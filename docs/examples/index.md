# Examples

Fifteen worked examples, each a problem stated precisely and solved end to
end. Each says when its method is the right tool, writes out the mathematics
beside the code, cites the papers the method comes from, and ends with
checks of what it claims. The {doc}`../tutorials/index` teach one idea at a
time; the {doc}`../user-guide/index` answers when and why; these
demonstrate.

Every example runs from a clean checkout with no data and no code of your
own, almost all of them on the problems of {doc}`../user-guide/toy-models`. They are
grouped by the level of the API they work at, as the user guide is.

## Running an algorithm

| | example | shows |
| --- | --- | --- |
| 1 | {doc}`ex01_eki` | EKI in its sampling and its optimization form, with one driver |
| 2 | {doc}`ex02_enkf` | the ensemble Kalman filter on Lorenz-96, in one call |
| 11 | {doc}`ex11_hybrid` | a filter whose update blends the sample covariance with a static one |
| 12 | {doc}`ex12_localization` | domain localization of both update rules, with ten particles for forty variables |

## One update

| | example | shows |
| --- | --- | --- |
| 3 | {doc}`ex03_one_update` | one update with each rule, against the exact posterior |
| 6 | {doc}`ex06_two_estimators` | why adding a known noise covariance beats sampling the noise |
| 14 | {doc}`ex14_custom_rule` | a new update rule, used unchanged in an update, a filter and an EKI run |

## Forecast and update separately

| | example | shows |
| --- | --- | --- |
| 4 | {doc}`ex04_hand_loop` | the EKI loop written as explicit calls, reproducing the driver |
| 9 | {doc}`ex09_gibbs` | a Gibbs sampler alternating an ensemble update with a conjugate one |
| 13 | {doc}`ex13_named_blocks` | estimating a parameter and a smoothed state as extra blocks of a filter |

## Probabilistic operations

| | example | shows |
| --- | --- | --- |
| 5 | {doc}`ex05_probabilistic_ops` | both update rules built from conditioning and conditional maps |
| 7 | {doc}`ex07_kalman_filter` | the exact Kalman filter and its log likelihood, from Gaussian operations alone |
| 8 | {doc}`ex08_hyperparameters` | fitting hyperparameters by the gradient of the log evidence |
| 10 | {doc}`ex10_importance` | EKI as the proposal of an importance sampler |
| 15 | {doc}`ex15_differentiability` | derivatives through an update, an EKI run, and an ensemble likelihood whose simulator is outside JAX |

```{toctree}
:hidden:
:maxdepth: 1

ex01_eki
ex02_enkf
ex03_one_update
ex04_hand_loop
ex05_probabilistic_ops
ex06_two_estimators
ex07_kalman_filter
ex08_hyperparameters
ex09_gibbs
ex10_importance
ex11_hybrid
ex12_localization
ex13_named_blocks
ex14_custom_rule
ex15_differentiability
```

## How the examples are built

Each example is a percent-format Python file in `docs/examples/src/`, which
is what is reviewed and edited. `docs/examples/build.py` executes one in a
fresh interpreter and writes the notebook beside it with its outputs stored;
the documentation renders those stored outputs and never executes anything.
To run one yourself, run its source, or open its notebook:

```bash
uv run python docs/examples/src/ex01_eki.py
```

To regenerate the notebooks after changing a source or the package:

```bash
uv run python docs/examples/build.py
```

Two tests keep them honest. A fast one checks that every stored notebook's
cells are exactly its source's, so a notebook cannot drift from the code it
claims to show. A slow one, `uv run pytest -m slow`, executes every source,
and each source ends with checks of the numbers its text states.

The examples on Lorenz-96 (2, 11, 12, 13, 14) print numbers that differ
from one platform to another. The system is chaotic, and its spin-up turns a
difference in the last digit of one arithmetic operation into a different
true trajectory, so another machine filters a different truth at the same
seed. Their text states approximate values, and their checks test bands
measured over several seeds.
