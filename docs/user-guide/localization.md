# Localization

An ensemble Kalman update estimates every covariance it needs from $J$
particles. When the targets and the given blocks are spatially extended and
much larger than $J$, two things go wrong at once. Every update moves a target
block within the span of its own anomalies, a space of dimension at most
$J - 1$, so most directions can never be corrected. And the sample covariance
between two coordinates far apart is noise of order $1/\sqrt{J}$ rather than
the near-zero it should be, so every datum moves every coordinate.
**Localization** fixes both by letting each coordinate see only the data near
it.

`kalman.LocalizedUpdateRule` implements **domain localization**, the scheme
of the local ensemble transform Kalman filter (Ott et al., 2004; Hunt et al.,
2007): each target coordinate gets its own update, from only the given
coordinates near it, each with its noise inflated by the reciprocal of a taper
of its distance. It wraps either shipped rule, and it is an `UpdateRule`, so
it goes wherever a rule goes. The alternative, covariance localization
(Houtekamer & Mitchell, 2001; Hamill et al., 2001), tapers the sample
covariance itself; the two are closely related (Sakov & Bertino, 2011), but
the taper's entrywise product destroys the low rank every update in EnsKit is
built on, so EnsKit provides domain localization only. The
{doc}`../kalman-contract` specifies it exactly, in its section
*Localization*.

## The problem

A field on 1000 sites of a line has a smooth Gaussian prior, with correlation
length 25 sites, and is observed at 40 sites with noise variance $r = 0.05$:

$$
x \sim \mathcal N(0, C), \quad C_{pq} = \exp\!\Big(-\frac{(p - q)^2}{2 \cdot 25^2}\Big),
\qquad y = Hx + e, \quad e \sim \mathcal N(0, r I_{40}),
$$

with $H$ selecting the observed sites. The problem is linear and Gaussian, so
the exact posterior mean $C H^\top (H C H^\top + r I)^{-1} y$ is available to
compare against. The prior covariance is built here, by the caller: EnsKit
never constructs one.

```python
import enskit  # enables float64; import this before creating arrays
import jax
import jax.numpy as jnp
import numpy as np
from enskit import kalman
from enskit.distribution import Ensemble
from enskit.linalg import PSDDiagonal

P, N, J, r = 1000, 40, 20, 0.05
sites = np.arange(P, dtype=float)
C = np.exp(-0.5 * ((sites[:, None] - sites[None, :]) / 25.0) ** 2) + 1e-6 * np.eye(P)
L = np.linalg.cholesky(C)
observed = np.linspace(10, P - 10, N).round().astype(int)

rng = np.random.default_rng(0)
truth = L @ rng.normal(size=P)
y = truth[observed] + np.sqrt(r) * rng.normal(size=N)
H = np.eye(P)[observed]
gain = C @ H.T @ np.linalg.inv(H @ C @ H.T + r * np.eye(N))
exact_mean = gain @ y
exact_spread = np.sqrt(np.diag(C - gain @ H @ C)).mean()

x = (L @ rng.normal(size=(P, J))).T          # J = 20 prior particles
ens = Ensemble(x=jnp.asarray(x), g=jnp.asarray(x[:, observed]))
noise = {"g": PSDDiagonal(jnp.full(N, r))}
```

The global square-root update, measured two ways: how far the posterior mean
is from the exact one, and what fraction of the change in the mean lies
outside the span of the prior anomalies.

```python
anomalies, _ = np.linalg.qr((x - x.mean(axis=0)).T)   # a basis of their span


def measure(post):
    particles = np.asarray(post["x"])
    mean = particles.mean(axis=0)
    change = mean - x.mean(axis=0)
    outside = change - anomalies @ (anomalies.T @ change)
    error = np.sqrt(np.mean((mean - exact_mean) ** 2))
    spread = particles.std(axis=0, ddof=1).mean()
    return error, np.linalg.norm(outside) / np.linalg.norm(change), spread


glob = kalman.update(ens, g=jnp.asarray(y), noise=noise,
                     update_rule=kalman.SymmetricSquareRoot())
error_global, outside_global, spread_global = measure(glob)
```

`outside_global` is zero to round-off: the update cannot leave a
19-dimensional subspace of a 1000-dimensional field. Its error,
`error_global`, is about 0.50, against a root-mean-square posterior mean of
about 0.83, and its spread `spread_global` is far too small, an average
standard deviation of 0.15 against the exact posterior's 0.21
(`exact_spread`): the familiar collapse of an unlocalized small ensemble.

## Localizing the update

A `DomainLocalization` says where every coordinate lives and how far its
influence reaches:

```python
localization = kalman.DomainLocalization(
    target_coords={"x": sites[:, None]},          # (1000, 1): one location per coordinate
    given_coords=sites[observed][:, None],        # (40, 1): where each datum lives
    radius=75.0,
    max_neighbors=6,
)
local_rule = kalman.LocalizedUpdateRule(kalman.SymmetricSquareRoot(), localization)
loc = kalman.update(ens, g=jnp.asarray(y), noise=noise, update_rule=local_rule)
error_local, outside_local, spread_local = measure(loc)
```

Coordinate $p$ is now updated from its 6 nearest data $i_1, \dots, i_6$
only, datum $i_k$ entering with noise variance $r/\rho_{pk}$, where

$$
\rho_{pk} = \mathrm{GC}\big(d(c_p, c_{i_k}) / L\big), \qquad L = 75,
$$

and $\mathrm{GC}$ is the Gaspari–Cohn taper (Gaspari & Cohn, 1999), equal to 1
at distance zero and exactly 0 from distance $L$ on. The error falls to about
0.07, seven times smaller, and the spread matches the exact posterior's.
Each coordinate has its own weights, so most of the change in the mean,
`outside_local`, now lies outside the prior anomalies' span.

## The geometry

**Target blocks.** `target_coords` maps every target block to a `(d, q)`
array of locations, one row per coordinate, in any number $q$ of dimensions.
A block that has no location, such as a parameter shared by the whole domain,
maps to `None` and gets the wrapped rule's ordinary global update, from every
datum. Tapering such a parameter would destroy the pooling over the whole
domain that makes it global. Every target block must be named: one left out
raises, rather than quietly getting the global update.

```python
both = kalman.DomainLocalization(
    target_coords={"x": sites[:, None], "offset": None},  # "offset" is global
    given_coords={"g": sites[observed][:, None]},
    radius=75.0, max_neighbors=6,
)
```

**Given blocks.** `given_coords` maps each given block to its locations. A
single bare array is accepted when one block is given, which is what you pass
to an algorithm whose given block is internal to it.

**Distance and taper.** The distance is Euclidean by default; a periodic
domain passes its own, `distance(point, points)` from a `(q,)` point and
`(m, q)` points to `(m,)` distances. The taper is `kalman.gaspari_cohn` by
default and may be any function of $r = d/L$ with values in $[0, 1]$. It need
not be a positive-definite function, unlike a taper for covariance
localization, since here it only scales noise variances.

```python
def ring(point, points, n=40):                 # sites 0, ..., 39 on a ring
    d = jnp.abs(points[:, 0] - point[0])
    return jnp.minimum(d, n - d)


ring_localization = kalman.DomainLocalization(
    target_coords={"x": np.arange(40.0)[:, None]},
    given_coords=np.arange(0.0, 40.0, 2.0)[:, None],
    radius=8.0, max_neighbors=10, distance=ring,
)
```

## Choosing the radius and the neighborhood size

The **radius** sets how far a datum's influence reaches. Too small and each
coordinate ignores data that do inform it; too large and the spurious
long-range correlations come back. A few correlation lengths of the field is
the usual starting point, tuned on a problem whose answer is known. The
neighborhood size **`max_neighbors`** fixes how many data each local update
can use, which keeps every local problem the same shape so that all of them
vectorize. Set it to at least the number of data within the radius: beyond
the $K$ nearest, data are cut off even inside the radius. Here the data are
about 25 sites apart, so at most 6 lie within 75 sites of any coordinate, and
`max_neighbors=6` cuts none off: a larger $K$ gives the same result. The cost of a build is one $J \times K$
singular value decomposition per located coordinate, computed once, so a call
afterwards only gathers and contracts.

## What it requires

- **Diagonal noise.** Each given block's noise must be a `PSDDiagonal` or an
  `Identity`, possibly scaled, which covers the tempered noise
  `R * (1 / delta)` of an inversion. A local problem needs the noise of its
  neighborhood alone, and a correlated block has no such restriction.
- **The default approximation.** The rule reads the given anomalies off the
  approximation's factor, so it takes `kalman.gaussian_approximation`, and
  refuses a modified approximation such as a hybrid covariance.
- **No target noise.** No target block may carry an independent term.

Around `kalman.Matheron`, every local update uses the same noise draw, and
with a key each coordinate's spread is that of its local problem. With no
localization at all (every datum in every neighborhood, at weight 1) either
localized rule gives exactly its wrapped rule's update.

## Inside an algorithm

A localized rule needs nothing from the code running it: pass it wherever an
update rule goes, to `kalman.update` as here, or to an algorithm's driver.

```python
post = kalman.update(ens, g=jnp.asarray(y), noise=noise,
                     update_rule=kalman.LocalizedUpdateRule(kalman.Matheron(), localization),
                     key=jax.random.key(0))
```

## References

- Gaspari, G. & Cohn, S. E. (1999). Construction of correlation functions in
  two and three dimensions. *Quarterly Journal of the Royal Meteorological
  Society*, 125(554), 723–757.
- Hamill, T. M., Whitaker, J. S. & Snyder, C. (2001). Distance-dependent
  filtering of background error covariance estimates in an ensemble Kalman
  filter. *Monthly Weather Review*, 129(11), 2776–2790.
- Houtekamer, P. L. & Mitchell, H. L. (2001). A sequential ensemble Kalman
  filter for atmospheric data assimilation. *Monthly Weather Review*, 129(1),
  123–137.
- Hunt, B. R., Kostelich, E. J. & Szunyogh, I. (2007). Efficient data
  assimilation for spatiotemporal chaos: a local ensemble transform Kalman
  filter. *Physica D*, 230(1–2), 112–126.
- Ott, E., Hunt, B. R., Szunyogh, I., Zimin, A. V., Kostelich, E. J.,
  Corazza, M., Kalnay, E., Patil, D. J. & Yorke, J. A. (2004). A local
  ensemble Kalman filter for atmospheric data assimilation. *Tellus A*,
  56(5), 415–428.
- Sakov, P. & Bertino, L. (2011). Relation between two common localisation
  methods for the EnKF. *Computational Geosciences*, 15(2), 225–237.
