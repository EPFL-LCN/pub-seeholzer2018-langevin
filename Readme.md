# CILES: Continuous Interval Langevin Equation Simulator

Langevin integrator for SDEs with constant drift and diffusion on continuous intervals with circular boundary conditions.

CILES is written in Cython and uses GSL for interpolation of drift & diffusion fields, to be able to simulate continuous variables.

## Description

Given a discretized drift field A(x) and a (position dependent) diffusion coefficient B(x) this tool performs simple time-forward integration of the SDE:

```
dx(t)/dt = A(x(t)) + sqrt(B(x(t))) * eta(t)
```

where eta(t) is a gaussian white noise term and x is a variable on an interval with circular boundaries (commonly 0 <= x < 2PI).

Both drift field A and diffusion B need to be arrays of the same dimension. They are internally interpolated (using ``gsl_interp_cspline_periodic``) to provide continuous fields, which are then used in the forward integration.

Forward integration is performed with the [Euler-Murayama scheme](https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method):
x(t+dt) = x(t) + dt * A(x(t)) + r * sqrt(dt * B(x(t))),
where r is a normally distributed random number with zero mean and unit variance.

## Dependencies
* Numpy
* Cython
* [Cython-gsl](https://github.com/twiecki/CythonGSL)

## Installation
* Clone repository
* `python setup.py install`
* To test: `python setup.py test`

## Example use

```python
from ciles.ciles import LangevinIntegrator as LI
import numpy as np

drift = np.zeros(100)  # no drift field
diff = np.ones(100)  # constant diffusion with 1 deg^2/s

dt = 1e-3  # 1 ms timestep
tmax = 1.  # simulate until 1s

# initialize the integrator
li = LI(drift, diff, dt=dt, tmax=tmax)

# simulate a single trajectory
li.run()
out = li.out
```

## More examples

See ``examples.py`` for more example usage.

### Final distributions after 2s diffusion
See the [source](https://github.com/flinz/ciles/blob/master/examples.py#L7):
![Diffusion for 2 seconds](https://user-images.githubusercontent.com/97735/33634816-ce92b380-da15-11e7-944c-e704cbe9cfab.png)

### Trajectories for drift-field with 2 fixed points
See the [source](https://github.com/flinz/ciles/blob/master/examples.py#L37):
![Plotting trajectories](https://user-images.githubusercontent.com/97735/33634815-ce790f48-da15-11e7-9cd9-1e08fdab9773.png)

