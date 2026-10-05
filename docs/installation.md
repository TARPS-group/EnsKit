# Installation

EnsKit needs Python 3.11 or later, and depends on JAX 0.10.1 or later and
NumPy 2.0 or later. It is not on PyPI yet; each release is a tag of the
repository, and installs from there.

## As a dependency

With [uv](https://docs.astral.sh/uv/):

```bash
uv add "enskit @ git+https://github.com/TARPS-group/EnsKit@v0.1.0"
```

which records the dependency in your `pyproject.toml` as

```toml
[project]
dependencies = ["enskit"]

[tool.uv.sources]
enskit = { git = "https://github.com/TARPS-group/EnsKit", rev = "v0.1.0" }
```

Only uv reads `tool.uv.sources`. For a `pyproject.toml` that pip and other
tools also read, write the dependency as a direct reference, which
`uv add --raw` records:

```toml
[project]
dependencies = ["enskit @ git+https://github.com/TARPS-group/EnsKit@v0.1.0"]
```

With pip:

```bash
pip install "enskit @ git+https://github.com/TARPS-group/EnsKit@v0.1.0"
```

Name the tag rather than a branch, so an install does not change under you.
The {doc}`changelog` lists what each release changes; until 1.0, a minor
release (0.2, 0.3, ...) may change the interfaces, and a patch release does
not.

## From a clone

To work on EnsKit itself, use uv:

```bash
git clone https://github.com/TARPS-group/EnsKit.git
cd EnsKit
uv sync --group dev
```

This creates a `.venv`, installs EnsKit in editable mode with its runtime
dependencies, and adds the test, lint and documentation tooling.

## Verifying the install

```python
import enskit
enskit.__version__     # '0.1.0'
```

From a clone, `uv run pytest` runs the test suite.

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
