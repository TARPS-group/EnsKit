# CLAUDE.md — EnsKit (formerly pyEKI)

## What this project is

EnsKit (package `enskit`) is a small, robust, efficient toolkit of building
blocks for ensemble Kalman methods: distributions over named blocks, maps
that push them forward, and ensemble Kalman updates, with Ensemble Kalman
Inversion (EKI) and the ensemble Kalman filter (EnKF) shipped as algorithms
built from those blocks. It serves two audiences: people who want EKI or an
EnKF to work, and people who build their own algorithms from the pieces.

It is a **library**, not a research repository. The deliverable is a
well-documented, well-tested package that colleagues can depend on. Prefer
clarity and correctness over cleverness, and keep the public surface small.

**The redesign is in progress.** `docs/redesign/index.md` records the design
and the plan, a sequence of pull requests numbered 0 to 12; the full design
document is `docs/redesign/design.html`, and `docs/redesign/stubs/` holds the
docstrings each layer starts from. PR 1 renamed the package from `pyeki`;
the GitHub repository is `TARPS-group/pyEKI` until it is renamed. Until PR 7,
the old `gauss` and `eki` modules exist beside the new layers and keep their
own contracts (`docs/gaussian-contract.md`, `docs/eki-contract.md`); do not
extend them, and do not make the new layers depend on them.

The layers, each building on the ones above it in this list:

1. `enskit.linalg` — structured linear operators, scoped to what ensemble
   Kalman methods need rather than to general-purpose linear algebra.
   `IdentityPlusGram` is the conditioning core.
2. `enskit.distribution` — `Ensemble`, `Gaussian`, `EnsembleGaussian`, the
   conditional maps `MatheronMap` and `SquareRootMap`, and weights.
3. `enskit.maps` and `enskit.kalman`, siblings that never import each other:
   pushing distributions through maps, and ensemble Kalman updates.
4. `enskit.algorithms` — the EKI and EnKF drivers and their policies.

Beside them, `enskit.toy` holds toy problems for this package's own tests and
its documentation, and `enskit.testing` holds conformance checks. Both may
import any layer; **nothing in the layers may import either**, which is what
keeps toy problems from becoming load-bearing.

## What this project is NOT

Out of scope, deliberately and permanently:

- **Forward models.** A simulator is any callable from a batch of inputs to
  a batch of outputs. EnsKit ships toy models for testing and documentation
  only, in `enskit.toy`. The optional wrappers in `enskit.maps` (`Linear`,
  `AdditiveNoise`, `BlackBox`) carry structure, not behavior. A toy model that
  wants a domain-specific name belongs in a calling repository.
- **Priors, Gaussian process kernels, coregionalization.** A prior is any
  operator satisfying the covariance interface. Constructing covariances from
  kernels belongs to the caller.
- **Domain-specific anything.** No knowledge of the systems being calibrated
  or filtered should appear in this package, including in docstrings and
  examples.
- **A general-purpose linear algebra library.** `enskit.linalg` exists because
  ensemble Kalman methods need structured operators. Add a structure when a
  method needs it, not because it would be nice to have.
- **A probabilistic programming language, distributed computing, MCMC
  kernels or an SMC driver.** EnsKit supplies the machinery of ensemble Kalman
  methods and interfaces with outside code through plain arrays, operators
  and distributions.

## Layer boundaries and vocabulary

Imports go **down only**: `distribution` imports `linalg`; `maps` and `kalman`
import `distribution` and `linalg`; `algorithms` imports all of them. An
import-linter contract in `pyproject.toml` enforces this in CI (from PR 2), and
`tests/test_toy.py` checks in a fresh interpreter that no layer loads
`enskit.toy`. Cross-layer chaining in user code goes through `pipe`
(`g.pipe(maps.pushforward, ...)`), never through an upward import. An
underscore module is imported only from inside its own package.

**Vocabulary flows downward only**, as imports do. A layer may name a layer
above it to justify its own scope; it may not borrow that layer's concepts to
define its own behavior.

| layer | speaks of | must not speak of |
| ----- | --------- | ----------------- |
| `linalg` | operators, batches, rows, factors, whiteners | distributions, blocks, particles, conditioning, anything above |
| `distribution` | distributions, blocks, particles, samples, weights, the latent space, means, covariances, sampling, marginals, conditioning, projection, realizing, conditional maps | maps, simulators, updates, observations, forecasts, steps, time, tempering |
| `maps` | maps, simulators, inputs, outputs, pushforward, noise | updates, gains, observations, steps, time, tempering |
| `kalman` | updates, update rules, given and target blocks, gains, transforms, localization, inflation | observations, forecasts, steps, runs, schedules, levels, time |
| `algorithms` | runs, steps, phases, levels, schedules, tempering, forecasts, analyses, observations, time | — |
| `toy` | anything above, to describe a problem | it may not *define* a schedule, an update rule, an inflation or a stopping rule |

"Observation operator" appears nowhere; below `algorithms`, a block becomes
a *given* block only at the moment it is conditioned on.

**One word per concept.** A *sample* is a draw from any distribution. A
*particle* is an element of an `Ensemble`; where samples are expected, an
ensemble's particles are those samples. Sizes are `n_particles`, `n_steps`,
`n_evaluations`, `latent_dim` and `dims[name]`. Blocks are named, so there are
no `u_dim`/`v_dim` in the layers (the toy problems take `parameter_dim` and
`data_dim`). EKI's terms are normative and specified in its contract under
*Terminology*: a **run** contains **steps**, each with two **phases**
(`evaluate` and `assimilate`) made of numbered **operations**, and each step
is preceded by one **evaluation** of the forward model. "Rung" and
"iteration" as a countable noun are retired.

## Pull requests and handoffs

The redesign is built one pull request at a time, each in its own fresh
session. Sessions do not carry state for each other: **the repository and its
GitHub issues are the source of truth.** Anything a later session needs goes
into one of these, never only into a chat or a prompt:

| what | where |
| --- | --- |
| the design and the plan | `docs/redesign/index.md`; the normative contracts in `docs/` as they are written, which supersede the design where they differ |
| a PR's scope, completion criteria and exceptions | its tracking issue in the "EnsKit 0.1" milestone |
| project rules | this file |
| current state, gotchas, notes for upcoming PRs | `HANDOFF.md` |
| a design tension with concrete instances | a new GitHub issue, with the numbers and the options |

**Starting a PR.**
1. Read this file, `docs/redesign/index.md`, the redesign section and
   "Next steps" of `HANDOFF.md`, and the tracking issue (`gh issue view N`).
2. Run `git fetch --all` and branch `dev/<topic>` off `origin/main`, never
   local `main`, which is often stale.
3. The issue defines the scope. If the issue and the docs disagree, or the
   scope is unclear, ask the maintainer before writing code.

**During a PR.**
- `uv run pytest`, `uv run ruff check .` and
  `uv run sphinx-build -b html -W docs docs/_build/html` all pass before every
  commit.
- Before opening the PR, run an adversarial review with a subagent. Split
  its findings: a design tension with concrete instances becomes an issue; a
  trivial defect is fixed in the PR.
- If the implementation must depart from the design, change the contract or
  the design page in the same PR and say so in the PR body.
  `docs/redesign/design.html` and the rest of `docs/redesign/` are a frozen
  record and are not edited.

**Finishing a PR.**
1. Open it against `main` in the milestone, with `Closes #N`, the checks run,
   and any exceptions listed. The maintainer reviews and merges; never merge.
2. In the same PR, update `HANDOFF.md`: the current state, and anything the
   next PRs need that their issues do not already say.
3. Comment on the issue of any later PR that a finding affects.
4. Hand off by offering the next PR, or PRs, whose dependencies are met as a
   suggested task (the desktop app's task chip, which the maintainer starts
   with "start with worktree"). If a dependency is still in review, the
   prompt says to start only after it merges. The prompt carries pointers
   only, because everything else is already in the repository:

   > Implement PR N of the EnsKit plan, tracking issue #M. Follow "Pull
   > requests and handoffs" in `CLAUDE.md`.

   Without the desktop app, the maintainer pastes the same line into a new
   session.

## Package management

`uv`. Use `uv sync` and `uv sync --group dev`; never `pip install` into the
environment. Run tests with `uv run pytest`.

## Docstring conventions

These apply to all docstrings, and strictly to module-, class-, and
public-function-level ones.

**Write for the person calling the code.** Lead with what the thing is and how
to use it. Explain behavior, arguments, return values, and errors, not the
reasoning that led to the implementation.

**State the mathematics.** When a function or class implements a
mathematical object or operation, its docstring states it precisely in a
`.. math::` block of LaTeX, defining every symbol. A prose description of a
formula is not a substitute for the formula.

**Use clear, precise language and no unnecessary jargon.** Prefer a plain
description over a compressed technical phrase. Do not editorialize about the
design: sentences like "the split is load-bearing rather than cosmetic" state a
low-level design judgment and do not belong at the top of an API.

**Organize with sections.** Use numpydoc headings (`Parameters`, `Returns`,
`Raises`, `Notes`, `References`) and tables when listing several classes or
functions. A reader should be able to skim the structure.

**Put design rationale in a `Notes` section, or leave it out.** Consequential
lower-level decisions are worth recording when they are non-obvious or easy to
undo by accident, but they go at the bottom under `Notes`, never in the opening
description. Extended rationale belongs in `docs/design.md`.

**Scope each level distinctly; do not repeat yourself.**
- *Module*: what the module provides, an index of its contents, and any
  convention shared across everything in it.
- *Class*: what this class represents and its parameters. Do not restate
  module-level conventions.
- *Method/function*: what this call does, its arguments and return value. Do
  not restate class-level context.

**Keep docstrings self-contained, and cite the literature.** A reader with only
the source must be able to follow a docstring; do not depend on anything
outside the repository to explain behavior. When a function or class
implements a method from a paper, cite the paper in a numpydoc `References`
section with the full reference. The citation records where the method comes
from; the docstring must still be complete without it. Cross-reference other
modules and classes within the package freely, using Sphinx roles
(`:class:`, `:mod:`, `:meth:`, `:func:`).

**American English** in code, docstrings and documentation: "centering",
"behavior", "neighbor", "modeling".

## Documentation

Sphinx with the furo theme, `myst-parser` for Markdown pages, and `napoleon`
for numpydoc-style docstring sections. Build with:

```bash
uv run sphinx-build -b html -W docs docs/_build/html
```

Every user-facing feature needs a place in the user guide, not only an API
entry. The user guide explains *when and why*; the API reference explains
*what*. The user guide is organized by level of abstraction: running an
algorithm, one update, forecast and update separately, probabilistic
operations, operators.

Examples are notebooks: each states its setup precisely, says when the method
it illustrates is the right tool, shows the mathematics alongside the code,
and cites the papers its methods come from. Wherever documentation or an
example describes a method from a paper, it cites the paper.

## Code conventions

**Public API first.** In every module the public classes and functions come
first, in the order of the module's index table; private helpers follow below
them. Module constants, public or private, are the exception: they go at the
top, after the imports and `__all__`, unless they refer to a class defined
below.

**Array shapes: leading batch axes, core operand shape trailing.** This is the
NumPy generalized-ufunc rule and what `vmap` produces. It applies everywhere,
not only in `linalg`. A block value is `(d,)`; an ensemble block is
`(n_particles, d)`, and the particle axis is a batch axis to `linalg`.
Distributions are unbatched pytrees; a family of them comes from `jax.vmap`.

**Block values may be passed as keywords** wherever a function takes a set of
them (`g.condition(y=y_obs)`, `Ensemble(x=x, theta=theta)`); the positional
mapping form is always available as well.

**Contract the trailing axis.** Never write `M @ x` in an operator
implementation: for arrays of two or more dimensions it contracts the
second-to-last axis, which silently returns a wrong answer when the operator is
square. Use `enskit.linalg.dense_matvec`.

**Fail loudly.** Unsupported operations raise rather than falling back to dense
linear algebra. Size guards raise before allocating. The only dense fallback
is the explicit, off-by-default `linalg.dense_fallback` context.

**Return JAX scalars, not Python floats.** Converting fails on a tracer under
`jit`, and on any complex intermediate.

**Factorize eagerly, at construction, and store the result.** A constructor
may compute from its arguments (`DensePSD(A)` runs the Cholesky), but
everything the operator needs must end up in its fields: pytree reconstruction
rebuilds operators from their stored fields alone, bypassing the constructor.
A factorization the caller already has is passed by keyword instead
(`DensePSD(L=L)`). Never cache a factorization lazily: a cache written inside
a traced function is discarded, so the operator silently re-factorizes on
every call.

**Randomness enters through typed keys.** Always-random functions take the key
first; sometimes-random ones take a keyword-only `key=`. A key is consumed
whole; callers split it.

**Every new operator gets `check_operator`.** The conformance suite catches the
batch-rank and square-root bugs that otherwise produce wrong numbers without
raising.

## JAX notes

- Float64 is enabled in the package's `__init__.py`. Worker processes do not
  inherit it; that needs `JAX_ENABLE_X64=1` in the environment.
- Operators and distributions are pytrees with data and metadata fields
  declared explicitly; unflatten bypasses the constructor, so validation runs
  only at genuine construction.
- Operators and distributions compare by identity and are never
  `static_argnums`.
- `shape` is a property, not a stored field, so it stays concrete under `jit`.
- JAX has no generalized `eigh`; use a Cholesky whitening reformulation.
- A plain SVD's derivative is `nan` at exactly repeated or exactly zero
  singular values. Conditioning goes through `IdentityPlusGram`, whose custom
  derivative rules stay finite there; do not differentiate through
  `jnp.linalg.svd` in new conditioning code.

## Testing

`pytest`, in `tests/`. Five kinds:

1. **Conformance**: every operator instance through `check_operator`; from
   PR 6, every update rule through `check_update_rule` and every conditional
   map through `check_conditional_map`.
2. **Targeted regression**: one test per bug class that produces wrong numbers
   without raising. These are the valuable ones; do not delete them as
   redundant. When a layer is reimplemented, its regression tests are ported,
   with a record of which old test became which new one.
3. **Exactness**: where a closed form exists, check against it rather than
   against a tolerance chosen to make the test pass.
4. **Counts**: whitenings per update, compilations per run. A cost regression
   passes every numerical test, so only a count catches it.
5. **Examples**: every example notebook executes, and its final checks pass.
