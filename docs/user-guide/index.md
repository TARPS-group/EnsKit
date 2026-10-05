# User guide

The user guide answers *when and why* to reach for each piece of EnsKit; the
contracts in the reference section say exactly *what* each one does. Its pages
are organized by level of abstraction, from running a whole algorithm down to
the operators everything else is built on. Each level is built from the one
below it, so start at the highest level that does what you need, and go down
a level when it does not.

## Running an algorithm

One call runs a whole algorithm: Ensemble Kalman Inversion to calibrate the
parameters of a forward model, or the ensemble Kalman filter to track the
state of a dynamical system. This is the level for most uses.

```{toctree}
:maxdepth: 1

running-an-inversion
filtering
writing-a-forward-model
toy-models
```

## One update

One ensemble Kalman update moves particles toward a conditional distribution:
the choice of update rule, the Gaussian approximation it uses, localization
when the ensemble is much smaller than the blocks it updates, and inflation.
Each step of each algorithm is one of these.

```{toctree}
:maxdepth: 1

updates
localization
```

## Forecast and update separately

The algorithms alternate an update with a pushforward, which runs a simulator
on every particle. Writing the alternation yourself runs an algorithm the
drivers do not, such as joint state and parameter estimation or a smoother.

```{toctree}
:maxdepth: 1

forecast-and-update
maps
```

## Probabilistic operations

Updates and pushforwards are built from operations on two distributions over
named blocks, an ensemble of particles and a Gaussian: sampling, fitting,
adding noise, conditioning, and the two maps that move particles to a
conditional.

```{toctree}
:maxdepth: 1

distributions
```

## Operators

Covariances, noise and linear maps are structured linear operators, which
are applied, solved against and factored without being stored as dense
matrices.

```{toctree}
:maxdepth: 1

quickstart
operators
writing-an-operator
```
