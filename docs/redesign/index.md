# The EnsKit redesign

pyEKI is being redesigned as **EnsKit** (package `enskit`): a toolkit of
building blocks for ensemble Kalman methods in general, with Ensemble Kalman
Inversion (EKI) and the ensemble Kalman filter (EnKF) shipped as algorithms
assembled from those blocks. This page records the decisions and the plan.
It is the reference for the pull requests that implement the redesign, and it
is superseded, layer by layer, by the normative contract each of those pull
requests writes.

The full design document, with the API of every public class and function
written out as docstrings and fifteen worked examples executed against a
prototype, is `docs/redesign/design.html` in the repository. Open it in a
browser; it is built from the material in this folder (see
[Working material](#working-material)).

## The design in five decisions

1. **Two distributions, both over named blocks.** An `Ensemble` is an
   empirical distribution of particles, optionally weighted. A `Gaussian`
   is a joint Gaussian held as a *shared factor* plus an *independent term*
   per block,

   $$
   x_b = m_b + F_b\,\xi + e_b, \qquad \xi \sim \mathcal N(0, I_k), \qquad
   e_b \sim \mathcal N(0, D_b),
   $$

   so that $\operatorname{cov}(x_a, x_b) = F_a F_b^\top$ for $a \ne b$ and
   $\operatorname{cov}(x_b, x_b) = F_b F_b^\top + D_b$. Blocks are named
   vectors (`"u"`, `"g"`, `"x_prev"`), so code never indexes into a
   concatenated vector, and conditioning takes any number of blocks.
2. **The bridge between them is particle alignment.** `Ensemble.project()`
   returns an `EnsembleGaussian`, a `Gaussian` subclass whose latent coordinate
   $j$ is particle $j$. Only that subclass can read its particles back out
   (`realize_particles`) and move them as a set (`square_root_map`). It
   replaces today's `GaussianJoint` and `EmpiricalJoint`.
3. **The two classic updates differ in the order of two operations.** The
   symmetric square-root update [Bishop et al. 2001; Hunt et al. 2007] is
   *realize after condition*; the stochastic update [Burgers et al. 1998;
   Wilson et al. 2021] is *transport after realize*. An update rule is a
   factory: it builds an update from the particles, a joint Gaussian
   approximation and the *names* of the given blocks, and the given *values*
   arrive when the update is called.
4. **Structure is exploited through the order of operations.** Known
   additive noise, a linear map and a static covariance are exact operations
   on the joint Gaussian, not flags on an update. Adding a known noise
   covariance after fitting, rather than sampling the noise first, gives a
   lower-variance estimate of the same joint. Localization wraps an update
   rule.
5. **Strict layers, with imports that only go down,** enforced in CI.

## Layers and import rules

| layer | provides | may import |
| --- | --- | --- |
| `enskit.linalg` | structured operators; `IdentityPlusGram`, the conditioning core | — |
| `enskit.distribution` | `Ensemble`, `Gaussian`, `EnsembleGaussian`, `MatheronMap`, `SquareRootMap`, weights | `linalg` |
| `enskit.maps` | `pushforward`, `Linear`, `AdditiveNoise`, `BlackBox` | `distribution`, `linalg` |
| `enskit.kalman` | `update`, `gaussian_approximation`, `Matheron`, `SymmetricSquareRoot`, `LocalizedUpdateRule`, inflation | `distribution`, `linalg` |
| `enskit.algorithms` | `eki`, `enkf`, shared inflation and relaxation policies | everything above |
| `enskit.toy`, `enskit.testing` | toy problems; conformance checks | anything; imported by nothing |

`maps` and `kalman` are siblings and never import each other. Cross-layer
chaining in user code goes through `pipe`, so `g.pipe(maps.pushforward, ...)`
needs no upward import. The rules are checked by an import-linter contract in
CI and by the existing fresh-interpreter test.

Vocabulary flows downward only: observations, forecasts, time and tempering
appear only in `algorithms`. A *sample* is a draw from any distribution; a
*particle* is an element of an `Ensemble`.

## The update, in three stages

```python
approx = kalman.gaussian_approximation(ens, {"g": R})      # 1. approximate
step = kalman.Matheron().build(ens, approx, ("g",))         # 2. build, without values
post = step(g=y_obs, key=key)                               # 3. call with the values

post = kalman.update(ens, g=y_obs, noise={"g": R},
                     update_rule=kalman.Matheron(), key=key)  # the same, in one call
```

A `MatheronMap` is *pointwise*: it moves any samples of the joint. A
`SquareRootMap` moves *the particle set it was built from*, as a whole. The
`approximation=` hook of `kalman.update` and both drivers carries a modified
joint (a hybrid covariance, for example) into every update.

## Conventions

- **Arrays**: leading batch axes, core shape trailing. A block value is
  `(d,)`; an ensemble block is `(n_particles, d)`, and the particle axis is a
  batch axis to `linalg`. Distributions are unbatched pytrees; families come
  from `jax.vmap`.
- **Block values as keywords** wherever a function takes a set of them:
  `g.condition(y=y_obs)`, `Ensemble(x=x, theta=theta)`. The mapping form
  stays available.
- **Keys**: typed keys only. Always-random functions take the key first;
  sometimes-random ones take a keyword-only `key=`. Drivers split keys with a
  fixed arity.
- **Raise, never fall back silently to dense algebra.** An opt-in
  `linalg.dense_fallback(max_n=...)` exists for prototyping and is off by
  default.
- **Differentiability**: everything below the algorithm layer is
  differentiable. `IdentityPlusGram` carries custom derivative rules that stay
  finite at exactly repeated and zero singular values, where a plain SVD's
  derivative is `nan`. Simulator outputs are constants unless the simulator
  is traceable; `maps.BlackBox` makes a host-side simulator safe under `jit`,
  `vmap` and `grad` with a zero derivative.
- **The numerical rules of today's contracts carry over unchanged**: one
  thin SVD of the whitened factor and never its Gram matrix; the identity
  completion of the square-root transform; centering before whitening;
  $J + 1$ whitenings per stochastic update; log-space weights; per-step noise
  $R/\Delta\beta$, never $R/\beta$.

## New project rules

Four rules join `CLAUDE.md`: mathematics stated in `.. math::` blocks in
docstrings; American English; public API first in every module, helpers
below; and citations for every method taken from the literature (in a numpydoc
`References` section for docstrings), which amends the self-containment rule
without replacing it.

## Implementation plan

One pull request at a time, with `main` always green. The new layers are built
beside the old ones, which are deleted once the new EKI driver replaces them.
Each new layer's contract is written and reviewed before its code.

| PR | content | depends on |
| --- | --- | --- |
| 0 | adopt the design: this page, `CLAUDE.md`, `HANDOFF.md`, tracking issues, the working material | — |
| 1 | mechanical renames: `pyeki` → `enskit`, American spelling, public API first, `SquareLinOp.n` → `dim` | 0 |
| 2 | `linalg`: `IdentityPlusGram` (today's `gauss` switches to it), `Zero`, `LowRankUpdate`, `dense_fallback`, import-linter | 1 |
| 3 | the distribution contract | 2 |
| 4 | `enskit.distribution` | 3 |
| 5 | `enskit.maps`, with the simulator contract | 4 |
| 6 | `enskit.kalman`: contract and code | 4 |
| 7 | `enskit.algorithms.eki` on the new layers; old `gauss` and `eki` deleted | 5, 6 |
| 8 | `enskit.algorithms.enkf`, `toy.lorenz96`, `toy.linear_state_space` | 7 |
| 9 | `LocalizedUpdateRule`, `DomainLocalization` | 6 |
| 10 | the `Kronecker` family | 2 |
| 11 | documentation: user guide by level, examples gallery, tutorials, API reference | 8, 9 |
| 12 | release 0.1.0 | 11 |

The critical path is 0 → 1 → 2 → 3 → 4 → 6 → 7 → 8 → 11 → 12; PRs 5, 9 and
10 can run alongside it.

## Working material

Everything the design was built from lives in this folder. None of it is
package code: Sphinx and ruff skip it, and pytest never collects it.

| path | what it is |
| --- | --- |
| `design.html` | the full design document |
| `document/` | its sections and builder: `python docs/redesign/document/build.py` |
| `stubs/` | the API of every public class and function, as docstrings; the starting point for each layer's implementation |
| `notebooks/src/` | the fifteen examples as notebook sources, with setup, mathematics and citations |
| `notebooks/out/` | the executed notebooks and their rendered fragments |
| `notebooks/references.py` | the bibliography, each entry checked against the publisher or arXiv record |
| `prototype/` | the throwaway prototype the examples ran against, with the review's regression checks |

The prototype imports today's `pyeki.linalg`, so it runs against a checkout of
the repository before PR 1's rename, with the prototype directory on
`PYTHONPATH`:

```bash
PYTHONPATH=docs/redesign/prototype uv run python docs/redesign/notebooks/build_notebooks.py
```

It is a record, not a starting point for the implementation. It skips the
validation tiers and returns dense matrices where the design returns
operators. Its examples become acceptance tests.

## References

- Bishop, C. H., Etherton, B. J. & Majumdar, S. J. (2001). Adaptive sampling
  with the ensemble transform Kalman filter. Part I: Theoretical aspects.
  *Monthly Weather Review*, 129(3), 420–436.
- Burgers, G., van Leeuwen, P. J. & Evensen, G. (1998). Analysis scheme in the
  ensemble Kalman filter. *Monthly Weather Review*, 126(6), 1719–1724.
- Hunt, B. R., Kostelich, E. J. & Szunyogh, I. (2007). Efficient data
  assimilation for spatiotemporal chaos: a local ensemble transform Kalman
  filter. *Physica D*, 230(1–2), 112–126.
- Wilson, J. T., Borovitskiy, V., Terenin, A., Mostowsky, P. & Deisenroth,
  M. P. (2021). Pathwise conditioning of Gaussian processes. *Journal of
  Machine Learning Research*, 22(105), 1–47.
