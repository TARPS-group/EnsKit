# Forward model requirements

EnsKit supplies everything in a run except the forward model. This page is
what that callable must satisfy, written for the person wrapping one. A
forward model is a *simulator* in the sense of `enskit.maps`, and the
normative statement of its obligations is the simulator contract,
{ref}`maps-simulators`; {doc}`maps` describes simulators in general, with
several inputs and outputs. There is nothing to subclass and nothing to
register: EnsKit ships no forward models for real use and defines no base
class, protocol or registry for one.

## The interface

```python
import enskit  # enables float64; import before creating arrays
import jax.numpy as jnp

times = jnp.array([0.5, 1.0, 2.0])

def forward(u):                           # (J, 2) in
    return u[:, 0:1] * jnp.exp(-u[:, 1:2] * times)   # (J, 3) out
```

That is a complete forward model. {doc}`toy-models` ships this model, at
twelve observation points and with its own prior and data, and two others
ready-made, for trying the library before wrapping your own code.

:::{important}
**`forward` receives every particle at once, not one particle.** It is called
once per step with a `(J, P)` array and returns `(J, N)`: one prediction per
particle, in the same order. A function written for a single parameter vector
is the most common mistake here, and it fails badly: depending on the
arithmetic it either raises from deep inside JAX without mentioning the
particles, or returns a shape the driver refuses.
:::

If your model is naturally written for one particle, wrap it:

```python
import jax

def one_particle(u):                      # (P,) -> (N,)
    return u[0] * jnp.exp(-u[1] * times)

forward = jax.vmap(one_particle)          # (J, P) -> (J, N)
```

`jax.vmap` works there because that model is JAX. For one that is not, such
as a subprocess, a scheduler submission or a legacy binary, the wrapper is an
ordinary Python loop, which is equally legal.

With several parameter blocks the forward model receives one `(J, d_b)` array
per block, positionally, in block order or in the order `inputs=` names them;
{doc}`running-an-inversion` shows an example. Everything below holds for each
of those arrays.

## The whole obligation

| what | requirement |
| --- | --- |
| **arguments** | one positional argument per parameter block: a concrete `jax.Array`, shape exactly `(J, d_b)`, in the particles' dtype (`float64` under the package default) |
| **return** | any array-like of shape `(J, N)`: a `jax.Array`, a NumPy array, or a nested Python list |
| **dtype** | the particles'; a narrower floating dtype is promoted with a warning; a wider one, an integer, boolean or complex one raises |
| **rows** | row `j` of the return depends only on row `j` of the arguments |
| **failure** | signaled by a **non-finite row**, never by an exception |
| **exceptions** | the callable owns its own; anything that escapes stops the run |

Nothing else; see [what is not required](#what-is-not-required).

## The arguments

Each is a **concrete** `jax.Array`, never a tracer. The driver loop is
ordinary Python rather than `jax.lax.scan` precisely so this holds, and it is
what makes an external model legal: you can branch on the values, print them,
write them to disk, or block on a process that reads them.

Each is exactly two-dimensional, particles down the leading axis and
parameters across the trailing one, and never has a further axis in front,
since a run carries one ensemble. Its dtype is the particles' dtype, `float64`
unless you have disabled JAX's x64 mode.

Being a `jax.Array`, an argument must be converted for any library that does
not speak JAX:

```python
import numpy as np

u = jnp.ones((4, 2))                      # stands in for the argument
particles = np.asarray(u)                 # zero-copy, and read-only
```

:::{warning}
`np.asarray` returns a **read-only view**, not a copy. Assigning into it raises
`ValueError: assignment destination is read-only` from wherever you wrote,
not at the conversion. Copy with `np.array` if you need to modify.
:::

With an inflation configured, the arguments are the **inflated** particles:
the ones actually evaluated, and the ones the step's `Evaluation` carries. A
wrapper caching evaluations by parameter value should key on what it was
handed.

## The return

Any array-like of shape `(J, N)`. A `jax.Array`, a NumPy array and a nested
Python list of the same numbers are equivalent and give bit-identical runs,
so a wrapper that builds rows in Python or reads them from a file with NumPy
need not convert. One value per particle is `(J, 1)`, not `(J,)`.

The dtype is read before any conversion. The particles' dtype is written as
is. A **narrower** floating dtype, in practice `float32`, is promoted to the
particles' dtype, with a warning at each evaluation (Python's default filter
shows it once). A **wider** one raises, naming the
forward model, because writing it would silently throw away digits it
computed; an integer, boolean or complex return raises too.

EnsKit enables `float64` because ensemble anomalies are formed by subtraction
and lose digits to cancellation, and a model that computes or reports in
single precision has already lost them before the array arrives; promoting
prevents a *second* loss in the update's arithmetic and nothing more. Return
`float64` where you can. Where you cannot, the run is still legitimate, and
the warning is telling you the price.

## Signaling failure

A particle has *failed* when its prediction row contains any non-finite
entry. That is the entire signal, and it puts one real obligation on the
wrapper:

:::{important}
**A model that may crash, time out, exit non-zero, or lose a worker must catch
that itself and return a non-finite row** for the affected particles. An
array can express a failed particle but not a raised exception, so an
exception escaping the callable propagates out of the driver and stops the
run, and every particle's result for that step is lost with it.
:::

By default a failed particle raises an `EKIError` that names it.
`eki.run(..., on_failure="repair")` instead moves the failed particles to the
valid particles' center and continues; {doc}`running-an-inversion` explains
what repair costs. Either way, fewer than two valid particles raises.

The signal cannot see finite nonsense: zeros, an initial condition, a
sentinel such as `-9999`. The sentinel is the dangerous case: it is finite, so
the particle counts as valid, and its enormous misfit reads to an adaptive
schedule as genuine disagreement among the particles, so the run stalls
rather than flagging it. Map such values to non-finite rows in your wrapper,
where the information exists.

## Row independence

**Row `j` of the return must depend only on row `j` of the arguments.** The
update fits a Gaussian to the pairs of parameters and predictions and
conditions with the resulting cross-covariance, so a model that normalizes
across the particles, or shares a mutable accumulator between rows, returns
something that is not a sample of the joint distribution at all. **No run
detects it**: the shapes are right and the numbers are finite.

This is the only requirement beyond the shapes and the failure signal. From
*outside* a run it is detectable, and worth checking once, with
`enskit.testing.check_simulator`:

```python
from enskit.testing import check_simulator

check_simulator(forward, 2, 3)            # parameter dims, prediction dims
```

It permutes the particles and evaluates a subset of them again, which
between them catch a model that is order-dependent across rows and one that
normalizes across the particles. Neither is sufficient alone: a symmetric
coupling survives a permutation, and only the subset comparison sees it. It
also checks the shape at two ensemble sizes, the dtype, and, unless you pass
`stochastic=True`, determinism. It calls the model five times, so point it at
a cheap configuration of an expensive one. Its arguments are the input
dimensions, one per parameter block (`check_simulator(f, (1, 1), 3)` for a
model taking two one-dimensional blocks), and the output dimension.

(what-is-not-required)=
## What is not required

Jittability, traceability, `vmap`-ability, pure JAX, or differentiability.
The model is never traced and never inspected; NumPy, SciPy, a C extension, a
subprocess, an HTTP call to a cluster queue are all legal. (To call such a
model inside `jax.jit` yourself, outside the driver, wrap it in
`enskit.maps.BlackBox`; {doc}`maps` explains when.)

**Determinism is not required either.** A stochastic simulator is a legitimate
forward model. It costs three things, none of which raises: the exactness
result for linear models becomes a statement about the model *including* its
noise; the extra prediction spread damps the gain, so the run under-fits; and
both adaptive schedules read that spread as disagreement and shorten their
increments, so it costs more evaluations. A simulator that draws its
randomness from a JAX key sets the attribute `needs_key = True` and receives
the step's key as its first argument, which makes a run reproducible. Side
effects, such as scratch files, job submissions or a process pool, are
likewise fine.

## A worked example: an external executable

A solver invoked as a subprocess, one particle at a time, which sometimes
fails. The first block stands in for the external code, deliberately plain,
since the thing it represents is not a Python library. Substitute your own.

```python
import pathlib, subprocess, sys, tempfile

WORKDIR = pathlib.Path(tempfile.mkdtemp())
SOLVER = WORKDIR / "solver.py"
SOLVER.write_text(
    "import sys, math\n"
    "u = [float(x) for x in open(sys.argv[1])]\n"
    "if u[1] < 0.0:\n"
    "    sys.exit('solver diverged: negative decay rate')\n"
    "with open(sys.argv[2], 'w') as out:\n"
    "    for t in (0.5, 1.0, 2.0):\n"
    "        out.write(repr(u[0] * math.exp(-u[1] * t)) + '\\n')\n"
)
```

The wrapper:

```python
DATA_DIM = 3

def forward(u):
    """Evaluate the external solver once per particle."""
    particles = np.asarray(u)                       # read-only view; only read
    predictions = np.full((particles.shape[0], DATA_DIM), np.nan)

    for j, particle in enumerate(particles):
        path_in = WORKDIR / f"in_{j}.txt"
        path_out = WORKDIR / f"out_{j}.txt"
        path_out.unlink(missing_ok=True)            # never read a stale result
        np.savetxt(path_in, particle)
        try:
            subprocess.run(
                [sys.executable, str(SOLVER), str(path_in), str(path_out)],
                check=True, capture_output=True, timeout=60,
            )
            row = np.loadtxt(path_out)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                OSError, ValueError):
            continue                                # leave the row non-finite
        if row.shape == (DATA_DIM,):
            predictions[j] = row

    return predictions
```

Three details matter. **The result starts as `nan`**, so a particle is valid
only if something wrote over its row: every path that produces no prediction
is already a correctly signaled failure, which is what keeps the `except`
clause short enough to be right. **The `except` names its own failures**
rather than catching everything, so a bug in the wrapper still reaches you.
**Stale outputs are deleted first**, so a solver that exits zero without
writing cannot return a previous step's answer.

Driving it is unremarkable. Because the prior puts mass on negative decay
rates, where the solver fails, the run opts in to repair:

```python
import jax
from enskit import kalman
from enskit.algorithms import eki
from enskit.distribution import Gaussian
from enskit.linalg import PSDDiagonal

truth = jnp.array([2.0, 0.7])
y = truth[0] * jnp.exp(-truth[1] * times) + jnp.array([0.02, -0.01, 0.015])
noise = PSDDiagonal(jnp.full(DATA_DIM, 0.01))
prior = Gaussian.independent(u=(jnp.array([1.0, 1.0]),
                                PSDDiagonal(jnp.array([1.0, 0.5]))))

state = eki.EKIState.from_prior(jax.random.key(0), prior, n_particles=32)
result = eki.run(state, forward, y, noise,
                 update_rule=kalman.Matheron(),
                 schedule=eki.AdaptiveESSSchedule(),
                 on_failure="repair")

result.mean("u")       # [2.0495, 0.7477] against a truth of [2.0, 0.7]
result.min_n_valid     # 29 of 32 at the worst step
```

`min_n_valid` of 29 is the wrapper working: at the first step three
particles drew a negative decay rate, the solver exited non-zero, and the
wrapper turned each into a `nan` row that the driver repaired, with a
warning. Without
`on_failure="repair"`, the same run raises an `EKIError` at the first step,
naming the failed particles.

## Common mistakes

| symptom | cause |
| --- | --- |
| a shape error from inside JAX naming neither `J` nor `P` | `forward` written for one particle; wrap it with `jax.vmap` |
| `returned shape (N,) ... expected (J, d_out)` | returning one prediction vector rather than `(J, N)` |
| `assignment destination is read-only` | writing into `np.asarray(u)`; copy with `np.array` |
| a `UserWarning` about a narrower dtype | the model reports in single precision |
| an `EKIError` naming failed particles | the model returned non-finite rows; fix the model, or pass `on_failure="repair"` |
| the run stops with a traceback from your solver | an exception escaped the callable; catch it, return a non-finite row |
| the run stalls at tiny increments, every particle valid | a sentinel fill value such as `-9999`; map it to `nan` |
| a wrong answer with nothing raised | rows coupled across particles, or the return's row order not matching the argument's; `check_simulator` catches both |

The last row's other cause is a missing transpose. With particles in rows the
contraction is `u @ G.T`; writing `u @ G` raises for a rectangular `G`, but
for a **square** one it silently returns the transposed model's predictions,
right shape and no error.
