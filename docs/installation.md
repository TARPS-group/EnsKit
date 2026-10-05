# Installation

EnsKit uses [uv](https://docs.astral.sh/uv/) for environment and dependency
management.

## From a clone

```bash
git clone https://github.com/TARPS-group/EnsKit.git
cd EnsKit
uv sync
```

This creates a `.venv` and installs EnsKit in editable mode with its runtime
dependencies, JAX and NumPy.

To include the test, lint and documentation tooling:

```bash
uv sync --group dev
```

## As a dependency

Until EnsKit is published, depend on a local checkout:

```toml
[project]
dependencies = ["enskit"]

[tool.uv.sources]
enskit = { path = "../EnsKit", editable = true }
```

## Verifying the install

```bash
uv run pytest
```

## Float64

EnsKit enables JAX's float64 mode on import. JAX defaults to float32, which is
not accurate enough for the conditioning arithmetic — ensemble anomalies are
formed by subtraction, and the resulting cancellation costs several digits.

Two consequences:

- **Import `enskit` before creating any array.** Arrays built beforehand stay
  float32 and are not promoted afterward.
- **Worker processes do not inherit the setting.** If forward-model evaluations
  run in a process pool, set `JAX_ENABLE_X64=1` in the environment instead of
  relying on the import.

```bash
export JAX_ENABLE_X64=1
```

## GPU

EnsKit depends on `jax` without pinning an accelerator build. To run on GPU,
install the appropriate JAX wheel for your platform following the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
Nothing in EnsKit assumes CPU.
