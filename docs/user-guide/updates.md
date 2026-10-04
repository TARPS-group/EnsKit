# One ensemble Kalman update

`enskit.kalman` moves an ensemble's particles toward a conditional
distribution. Given particles over some *target* blocks and some *given*
blocks, and a value $y^*$ for the given blocks, an update returns particles
that approximate the conditional of the targets at $y^*$. This page is about
one such update: when to use each rule, how the approximation enters, and how
to write a rule of your own. Running many updates in a loop is the algorithm
layer's job.

The {doc}`../kalman-contract` specifies exactly what each call does: shapes,
errors, costs and the pinned random draws. The probabilistic operations the
updates are assembled from are on the {doc}`distributions` page.

## One call

```python
import enskit  # enables float64; import this before creating arrays
import jax
from enskit import kalman

# ens: an Ensemble with blocks "u" (what to update) and "g" (its prediction of y)
post = kalman.update(ens, g=y, noise={"g": R},
                     update_rule=kalman.Matheron(), key=jax.random.key(0))
```

`post` is an `Ensemble` over `"u"` alone: the given block is dropped. With
`noise`, the value `y` is read as a noisy realization of `g`,
$y = g + e$ with $e \sim \mathcal N(0, R)$, while `ens["g"]` holds the
noise-free predictions. The noise covariance is any `PSDLinOp`, and it is only
ever *whitened*, so a diagonal or Kronecker-structured covariance stays cheap.

There is no default rule: `update_rule` is required, because the two shipped
rules answer different questions.

## Which rule

**`SymmetricSquareRoot()`**, the ensemble transform Kalman filter's update
(Bishop et al., 2001; Hunt et al., 2007), is deterministic. It conditions the
ensemble's Gaussian approximation and reads the particles back out of the
conditioned Gaussian:

$$
x_j' = m_x + K(y^* - m_g) + \sqrt{J-1}\,F_x T e_j,
\qquad T = (I + SS^\top)^{-1/2},
$$

where $F_x$ holds the target's scaled anomalies and $S$ the whitened given
anomalies. For a linear problem its particles have exactly the conditional's
mean and covariance whenever the input particles have the prior's. Use it when
moments matter most and the problem is close to linear, or when you need a
deterministic update.

**`Matheron()`**, the stochastic or perturbed-values update (Burgers et al.,
1998; Houtekamer & Mitchell, 1998), moves each particle separately:

$$
x_j' = x_j + K\big(y^* - g_j - e_j\big), \qquad e_j \sim \mathcal N(0, R),
$$

with a fresh noise draw per particle, so it needs a key. Its moments are right
only in expectation over the key, with sampling error of order $1/\sqrt{J}$,
but because it redraws a perturbation for every particle it keeps the shape of
a nonlinear posterior better: the square-root update can leave a few particles
carrying most of the spread, and this one does not. It is Matheron's rule of
pathwise conditioning (Wilson et al., 2021) applied to particles.

Both rules use $K$ only through a whitened factor and one thin singular value
decomposition; neither forms a gain matrix or a data-space covariance.

## Three stages

`update` is shorthand for three stages, which you can run yourself when you
need one of them:

```python
approx = kalman.gaussian_approximation(ens, {"g": R})   # 1. approximate
built = kalman.Matheron().build(ens, approx, ("g",))     # 2. build, no values
post_a = built(g=y_a, key=k1)                            # 3. call with values
post_b = built(g=y_b, key=k2)                            #    ... as often as needed
```

1. **Approximate.** `gaussian_approximation` is the Gaussian with the
   particles' mean and covariance, jointly over every block, with the noise
   added to `g` as a *covariance*. Adding the known noise exactly, rather than
   adding sampled noise to the particles first, estimates the same joint with
   less variance.
2. **Build.** The rule computes everything that does not depend on $y^*$: the
   whitened factor, its decomposition, the square-root transform. This is
   where the cost is.
3. **Call.** Each call whitens one vector per given block and applies the
   result. Building once and calling at several values, or under `jax.vmap`
   over values, reuses the build.

## A different approximation

The approximation is where a modeling choice enters, and `update` takes it as
a function `(ensemble, noise) -> Gaussian`:

```python
import math
from enskit.distribution import Gaussian
from enskit.linalg import Dense, product

def hybrid(ensemble, noise, alpha=0.2):
    """alpha * sample covariance + (1 - alpha) * B on "x", and "y" = H x."""
    fit = ensemble.marginal("x").project()
    blend = Gaussian({"x": fit.mean("x")},
                     factors={"x": fit.factor("x") * math.sqrt(alpha)},
                     block_covs={"x": B * (1 - alpha)}).absorb("x")
    Fx = blend.factor("x")                  # [sqrt(alpha) X, sqrt(1 - alpha) L_B]
    joint = Gaussian({"x": blend.mean("x"), "y": H @ blend.mean("x")},
                     factors={"x": Fx, "y": product(Dense(H), Fx)})
    return joint.add_noise(noise)

post = kalman.update(ens, y=y, noise={"y": R}, update_rule=kalman.Matheron(),
                     approximation=hybrid, key=key)
```

A static covariance $B$ blended with the sample covariance, as in a hybrid
filter (Hamill & Snyder, 2000), fills in directions a small ensemble cannot
represent. The `absorb` matters: it moves $B$ into the shared factor, so that
it correlates `x` with `y` and enters the gain. Left as `x`'s independent term,
$B$ would be variation independent of `y`, and `Matheron` would add a draw of
it to every particle, which is additive inflation rather than a hybrid
covariance. `Matheron` works with any approximation. `SymmetricSquareRoot`
needs the default one, because it reads the particles back out of the
approximation's own factor, which only an approximation *aligned* with the
particles can do.

Alignment is a promise about values: the approximation is an
`EnsembleGaussian` whose latent coordinate $j$ is particle $j$.
`gaussian_approximation` keeps that promise. An `EnsembleGaussian` built from
other particles, or with its factor rescaled, breaks it silently, and inside
`enskit.linalg.debug_checks()` both rules check it. To inflate the spread, use
`inflate_multiplicative` on the particles before the update, not a rescaled
approximation.

## Exact values

Without `noise`, the given blocks are conditioned on exactly. Only
`SymmetricSquareRoot` supports that, and it needs more particles than the given
blocks have dimensions, since the conditional of an ensemble's Gaussian on $N$
exact values exists only when its anomalies span them:

```python
post = kalman.update(ens, g=y, update_rule=kalman.SymmetricSquareRoot())
```

## Failed and weighted particles

Updates take unweighted, finite particles. A weighted ensemble is refused;
resample it first with `enskit.distribution.resample`. A failed particle, a
row with a non-finite entry, must be repaired or dropped before the update.
Dropping is a reweight and a resample:

```python
from enskit.distribution import resample, reweight

ens = resample(key, reweight(ens, jnp.where(ens.all_finite, 0.0, -jnp.inf)))
```

Inside `debug_checks()` an update with a failed particle raises and says this;
outside it, every updated particle is `nan`.

## Inflation and relaxation

A finite ensemble underestimates its own spread, and four functions on
ensembles correct for that around an update:

| function | does |
| -------- | ---- |
| `inflate_multiplicative(ens, anomaly_scale)` | $x_j \mapsto \bar x + \lambda(x_j - \bar x)$: the covariance grows by $\lambda^2$ (Anderson & Anderson, 1999) |
| `inflate_additive(key, ens, x=Q)` | adds centered draws from $\mathcal N(0, Q)$: new directions, outside the span of the anomalies (Hamill & Whitaker, 2005) |
| `relax_to_prior_spread(prior, post, alpha)` | RTPS: scale each coordinate's posterior anomalies toward the prior's spread (Whitaker & Hamill, 2012) |
| `relax_to_prior_perturbations(prior, post, alpha)` | RTPP: blend posterior and prior anomalies particle by particle (Zhang et al., 2004) |

`anomaly_scale` multiplies the anomalies, not the covariance: a variance
inflation of 1.2 is `anomaly_scale=math.sqrt(1.2)`. The relaxations take the
ensemble the update started from and the update's result, whose particles
correspond one to one.

## Writing an update rule

An update rule is any class with a `build` method. It receives the particles,
their approximation and the given blocks' names, and returns a function of the
values. Here is the deterministic EnKF (Sakov & Oke, 2008), which gives the
mean the full Kalman correction and the anomalies half of it:

```python
class DEnKF:
    def build(self, particles, approximation, given):
        cmap = approximation.conditional_map(given)        # value-free
        halfway = particles.assign(
            {c: particles.mean(c) + 0.5 * particles.anomalies(c) for c in given})

        def update(values=None, /, *, key=None, **block_values):
            return cmap(halfway, values, **block_values).marginal(*cmap.targets)

        return update

post = kalman.update(ens, g=y, noise={"g": R}, update_rule=DEnKF())
```

It is then accepted everywhere a shipped rule is. Check it with
`enskit.testing.check_update_rule`, which builds a linear-Gaussian problem
with a known answer and checks the result's structure, that the build does not
depend on the values, the mean (and covariance, when the rule claims an exact
one), keys, and `jit` and `vmap`:

```python
from enskit.testing import check_update_rule

check_update_rule(DEnKF(), exact_covariance=False)   # its covariance is larger
```

`enskit.testing.check_conditional_map` does the same for a
`ConditionalMap`, against exact Gaussian conditioning.

## References

- Anderson, J. L. & Anderson, S. L. (1999). A Monte Carlo implementation of
  the nonlinear filtering problem to produce ensemble assimilations and
  forecasts. *Monthly Weather Review*, 127(12), 2741–2758.
- Bishop, C. H., Etherton, B. J. & Majumdar, S. J. (2001). Adaptive sampling
  with the ensemble transform Kalman filter. Part I: Theoretical aspects.
  *Monthly Weather Review*, 129(3), 420–436.
- Burgers, G., van Leeuwen, P. J. & Evensen, G. (1998). Analysis scheme in the
  ensemble Kalman filter. *Monthly Weather Review*, 126(6), 1719–1724.
- Hamill, T. M. & Snyder, C. (2000). A hybrid ensemble Kalman filter–3D
  variational analysis scheme. *Monthly Weather Review*, 128(8), 2905–2919.
- Hamill, T. M. & Whitaker, J. S. (2005). Accounting for the error due to
  unresolved scales in ensemble data assimilation: a comparison of different
  approaches. *Monthly Weather Review*, 133(11), 3132–3147.
- Houtekamer, P. L. & Mitchell, H. L. (1998). Data assimilation using an
  ensemble Kalman filter technique. *Monthly Weather Review*, 126(3), 796–811.
- Hunt, B. R., Kostelich, E. J. & Szunyogh, I. (2007). Efficient data
  assimilation for spatiotemporal chaos: a local ensemble transform Kalman
  filter. *Physica D*, 230(1–2), 112–126.
- Sakov, P. & Oke, P. R. (2008). A deterministic formulation of the ensemble
  Kalman filter: an alternative to ensemble square root filters. *Tellus A*,
  60(2), 361–371.
- Whitaker, J. S. & Hamill, T. M. (2012). Evaluating methods to account for
  system errors in ensemble data assimilation. *Monthly Weather Review*,
  140(9), 3078–3089.
- Wilson, J. T., Borovitskiy, V., Terenin, A., Mostowsky, P. & Deisenroth,
  M. P. (2021). Pathwise conditioning of Gaussian processes. *Journal of
  Machine Learning Research*, 22(105), 1–47.
- Zhang, F., Snyder, C. & Sun, J. (2004). Impacts of initial estimate and
  observation availability on convective-scale data assimilation with an
  ensemble Kalman filter. *Monthly Weather Review*, 132(5), 1238–1253.
