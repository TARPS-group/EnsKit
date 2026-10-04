# Operator catalog

## The hierarchy

Operators form three levels. Each level is a mathematical claim about the
map, and adds the operations that claim makes well defined — so an operator
never advertises something meaningless: a rectangular matrix has no `solve`
at all, rather than a `solve` that raises.

| class | represents | adds |
| --- | --- | --- |
| `LinOp` | any linear map, possibly rectangular | `matvec`, `rmatvec`, `matmat`, `rmatmat`, `T`, `to_dense` |
| `SquareLinOp` | a square map | `solve`, `solve_mat`, `logdet`, `diag` |
| `PSDLinOp` | a symmetric positive semi-definite map | `factor`, `whiten`, `whiten_mat` |

`rmatvec` applies the transpose, and `op.T` returns the transpose as an
operator. The full behavioral specification is the
{doc}`../linop-contract` reference page.

## Operators defined by their own arrays

| class | represents | notes |
| --- | --- | --- |
| `Identity(size)` | $I_n$ | every operation is free |
| `Zero(n_out, n_in)` | the zero matrix | stores no array; composites skip it |
| `PSDDiagonal(diagonal)` | $\mathrm{diag}(d)$ | all operations linear in $n$ |
| `Dense(A)` | an explicit array | may be rectangular; no structure assumed |
| `DenseSquare(A)` | a dense square matrix | stored with its LU; what `densify` returns for square non-PSD operators |
| `Triangular(L, lower)` | a triangular matrix | what `DensePSD.factor()` returns |
| `DensePSD(A)` | a dense PSD matrix | stored as its Cholesky factor; `DensePSD(L=L)` from a factor |
| `PSDLowRank(F)` | $FF^\top$ for a factor $F$ of shape $(n, k)$ | singular when $k < n$; provides `diag` and `factor` only |

`PSDLowRank` imposes no relation between $n$ and $k$, and computes nothing
at construction: the stored factor *is* the factorization, so `factor()`
hands it straight back as a `Dense`. It withholds `solve`, `whiten` and
`logdet` at *every* width. At $k < n$ that is forced — the operator is
singular by construction. At $k \ge n$ it holds because a capability
promises a *cheap* operation, and none of the three is cheap here: each
needs $FF^\top$ formed and factorized, which is what densifying does
anyway.

So densify when you need them — but only on an instance you know to be
full rank. `densify` of a thin-factor `PSDLowRank` takes the Cholesky of a
singular matrix, which returns `nan` with no exception unless you are
inside `debug_checks()`.

`DensePSD(A)` and `DenseSquare(A)` factorize the matrix once, at
construction. If you already have the factorization, pass it by keyword
instead — `DensePSD(L=L)` for a lower Cholesky factor,
`DenseSquare(A, lu=lu, piv=piv)` for an LU — and it is stored as given.
Operators never factorize lazily on first use, because a factor cached
inside a traced function is discarded when the trace ends, which would
silently re-factorize on every call.

## Operators built from other operators

Build composites through the factory functions, which pick the most capable
class for the ingredients:

| factory | returns | represents |
| --- | --- | --- |
| `block_diag(*blocks)` | `PSDBlockDiag` if every block is PSD, else `BlockDiag` | a block-diagonal matrix |
| `product(*ops)` | `Product` | $A_1 A_2 \cdots A_m$, applied right to left |
| `hstack(*ops)` | `HStack` | $[A_1\ A_2\ \cdots\ A_m]$, a block *row* |
| `diag_congruence(op, scale)` | `PSDDiagCongruence` | $\mathrm{diag}(s)\,A\,\mathrm{diag}(s)$ for PSD $A$ |

`HStack` splits its input along the trailing axis and sums the blocks'
outputs, so $[A_1\ A_2]\,[x_1; x_2] = A_1 x_1 + A_2 x_2$; its transpose
`hstack(...).T` is the corresponding block column. The block-diagonal
classes expose their children as `op.blocks` and their shapes as
`op.block_shapes`.

Pad an operator with zero columns using `Zero`, which stores no array:
`hstack(F, Zero(n, m))` is $[F\ \ 0]$, and applying it applies `F` alone.
A `Product` with a `Zero` factor applies as zero without applying its other
factors.

## A PSD operator plus a low-rank term

`LowRankUpdate(base, F)` represents $C = D + FF^\top$ for a PSD operator
$D$ that supports `whiten` and any operator $F$ of shape $(n, k)$. Reach for
it when a covariance is a cheap, invertible part plus a few directions of
extra variance — a diagonal noise covariance plus an ensemble's anomalies,
say — and you need to solve against it, whiten with it or take its
log-determinant without forming the $n \times n$ matrix:

```python
C = LowRankUpdate(PSDDiagonal(d), Dense(F))   # D + F F^T
x = C.solve(b)        # Woodbury: two applications of D's solve and whitener
w = C.whiten(y)       # W_C with W_C C W_C^T = I
ld = C.logdet()       # log det D + sum log(1 + sigma_i^2)
```

The constructor whitens $F$ by $D$ once, $S = (W_D F)^\top$, and stores the
`IdentityPlusGram` of $S$ (next section), so every later call reuses one
SVD. Its capabilities follow `base`: it can `solve` if `base` can, and so on
for `logdet`, `diag` and `factor`; `whiten` always works.

## The conditioning core: `IdentityPlusGram`

`IdentityPlusGram(S)` is the operator $A = I_k + SS^\top$ for a `(k, N)`
array $S$. Every Gaussian conditioning in EnsKit reduces to it: with $S$ the
transpose of a factor whitened by the noise, the Kalman gain applied to a
whitened residual is `A.solve_factor(r)` $= A^{-1}Sr$, and the conditioned
factor is the old one multiplied by `A.inverse_sqrt()` $= A^{-1/2}$. You need
it directly only if you are writing a conditioning of your own; the
distributions and updates use it for you.

```python
A = IdentityPlusGram(S)          # one thin SVD of S, at construction
w = A.solve_factor(r)            # (..., N) -> (..., k)
T = A.inverse_sqrt()             # an operator; T.to_dense() is (k, k)
```

Everything is computed from the thin SVD $S = U\Sigma V^\top$, never from
$SS^\top$, which would square the condition number. The multipliers of
`solve_factor` are $\sigma_i/(1+\sigma_i^2) \le 1/2$, so it stays bounded
however collapsed or large $S$ becomes.

**Derivatives.** A plain SVD's derivative divides by differences of singular
values, so it is `nan` whenever two singular values are exactly equal or
exactly zero — which happens routinely here: a localization mask zeroes
columns of $S$, and an exactly collapsed ensemble zeroes all of it.
`IdentityPlusGram`'s operations carry their own derivative rules, written in
terms of $S$ with no such division, so `jax.grad` through `solve`,
`solve_factor`, `logdet`, `whiten` and `inverse_sqrt` is finite and correct
at every $S$. Second derivatives are correct where the singular values are
distinct and nonzero; at degenerate spectra only `logdet`'s are promised.
The operator contract ({ref}`contract-gram`) states the rules.

## Prototyping with a dense fallback

An operation an operator cannot do cheaply raises `UnsupportedOpError`. While
prototyping on small problems you may prefer it to just work:

```python
from enskit.linalg import dense_fallback

with dense_fallback(max_n=2048):
    x = cov.solve(b)    # densifies cov if it has no cheap solve, with a warning
```

Inside the context an unsupported operation is computed on `densify(op)`,
with one warning per operator type and operation; an operator larger than
`max_n` still raises. `supports()` is unchanged, so code that checks it keeps
its structured branch. Each call densifies again at $O(n^3)$, and under `jit`
the choice is made when the function is traced, so a function compiled
inside the context keeps the dense path afterwards. Remove the context before
running anything large.

## Operator arithmetic

```python
C = A @ B            # composition: product(A, B)
R_t = R / dbeta      # tempered noise: a scaled operator, same capabilities
S = 2.0 * A          # scaling preserves the hierarchy level
At = A.T             # the transpose, as an operator
```

`@` composes **operators only**. Applying an operator to an array is always
`matvec`/`matmat` — `op @ x` raises a `TypeError` that says so, because
NumPy's `@` contracts axis `-2`, which is silently wrong for the
leading-batch vector layout everything in EnsKit uses. Scalars for `*` and
`/` may be traced values, which is what tempering needs.

(operator-batches)=
## Batches of operators

A batch of operators — one covariance per ensemble member, say — is built
with `jax.vmap` over the constructor, and used with `jax.vmap` over the
operator argument:

```python
covs = jax.vmap(DensePSD)(As)                      # As: (100, n, n)
outs = jax.vmap(lambda C, x: C.solve(x))(covs, xs) # xs: (100, n)
```

What `vmap` hands back is a *vmapped family*: a single operator object
whose stored arrays carry an extra leading axis. A family identifies
itself — `covs.batch_shape` is `(100,)` and its repr reads
`vmapped(DensePSD(3, 3), batch=(100,))` — and it is deliberately **inert**:
calling any operation on it directly, or scaling or composing it, raises a
`ValueError` telling you to apply it under `jax.vmap`, because outside of
`vmap` there is no defined way to line its members up with your data.
Passing arrays with extra leading axes to a constructor does *not* build a
family; it is rejected outright.

## Debugging value preconditions

Some requirements are about values, not shapes: `PSDDiagonal` entries must
be positive, `DensePSD` needs a symmetric positive-definite
matrix or, as `L=`, a genuine Cholesky factor. JAX cannot check values inside `jit`, so by default a violation
produces `nan` or `inf` downstream rather than an error. When a `nan`
appears and you want to find where, turn on debug checks:

```python
from enskit.linalg import debug_checks

with debug_checks():
    cov = DensePSD(A)   # raises here if A is not symmetric PD
```

`set_debug_checks(True)` enables them process-wide. The checks run only on
concrete arrays and are skipped on traced values, so enabling them never
changes `jit`-ed behavior.

## Conditional support

A composite's capabilities depend on its contents. A block-diagonal operator
supports `solve` only if every block does, so check before calling:

```python
if cov.supports("solve"):
    x = cov.solve(b)
else:
    x = densify(cov).solve(b)
```

`op.capabilities()` returns everything an operator supports beyond the
always-available operations. An unknown name raises `ValueError`, so a typo
cannot silently steer you onto the dense branch. For quick experiments,
`dense_fallback` (above) makes the dense branch implicit.

## Square roots and whitening

Two related operations, with different guarantees:

- **`factor()`** returns an operator `L` with `L @ L.T == op`, of shape
  `(n, k)`. Use it to draw samples: `L.matvec(eps)` for standard normal
  `eps` of length `k` has covariance `op`, and `L.rmatvec` applies `L.T`.
- **`whiten(x)`** applies a fixed matrix `W` with `W @ op @ W.T == I`, so
  data with covariance `op` becomes uncorrelated with unit variance.
  `whiten_mat` is the matrix-operand form.

The shape of `factor()` is informative. `k > n` means the operator is a sum
of simpler pieces, as in low-rank-plus-diagonal. `k < n` means it is
genuinely singular, so it supports neither `solve` nor `whiten` —
`PSDLowRank` is the shipped operator in that position.

The whitener is *not* promised to invert the factor: `whiten(L.matvec(eps))`
agrees with `eps` in distribution, never elementwise. Use one representation
per random draw.

## Cost summary

For an operator of side $n$:

| operator | `matvec` | `solve` | `whiten` | `logdet` |
| --- | --- | --- | --- | --- |
| `Identity` | $O(n)$ | $O(n)$ | $O(n)$ | $O(1)$ |
| `PSDDiagonal` | $O(n)$ | $O(n)$ | $O(n)$ | $O(n)$ |
| `DensePSD`, `DenseSquare` | $O(n^2)$ | $O(n^2)$ | $O(n^2)$ | $O(n)$ after the constructor's $O(n^3)$ |
| `Triangular` | $O(n^2)$ | $O(n^2)$ | — | $O(n)$ |
| `PSDLowRank` (factor width $k$) | $O(nk)$ | — | — | — |
| `Zero` | $O(1)$ work, $O(n)$ output | — | — | — |
| `IdentityPlusGram` ($S$ of shape $(k, N)$, $r = \min(k, N)$) | $O(kN)$ | $O(kr)$ | $O(kr)$ | $O(r)$ after the constructor's $O(kNr)$ |
| `LowRankUpdate` ($F$ of width $k$) | base $+ O(nk)$ | base solve and whiten $+ O(nk)$ | base whiten $+ O(nr)$ | base $+ O(r)$ |
| block diagonals | sum over blocks | sum over blocks | sum over blocks | sum over blocks |
| `PSDDiagCongruence`, scaled operators | base $+ O(n)$ | base $+ O(n)$ | base $+ O(n)$ | base $+ O(n)$ |
| `Product`, `HStack` | sum over factors | — | — | — |
