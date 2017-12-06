import numpy as np
cimport numpy as np
cimport cython
from cython_gsl cimport gsl_spline_eval, gsl_interp_accel, gsl_spline_alloc, gsl_spline, gsl_interp_cspline_periodic, gsl_interp_accel_alloc, gsl_spline_init,gsl_interp_accel_free,gsl_spline_free


# Cython compile-time declarations
TYPE_FLOAT = np.float64
ctypedef np.float64_t TYPE_FLOAT_t
ctypedef np.int_t TYPE_INT_t
ctypedef char TYPE_BOOL_t

# import c stdlib functions
cdef extern from "stdlib.h":
    ctypedef unsigned long size_t
    void free(void *ptr)
    void *malloc(size_t size)
    void *realloc(void *ptr, size_t size)
    size_t strlen(char *s)
    char *strcpy(char *dest, char *src)
    int c_rand "rand" ()  # rand in C, c_rand in Cython
    void c_srand "srand" (unsigned int seed)
    enum: RAND_MAX 
    double random_num "drand48" ()
    void srand48(long int seedval)

# import c math functions
cdef extern from "math.h":
    TYPE_FLOAT_t pow(TYPE_FLOAT_t base, TYPE_FLOAT_t exp)
    TYPE_FLOAT_t exp(TYPE_FLOAT_t x)
    TYPE_FLOAT_t tanh(TYPE_FLOAT_t x)
    TYPE_FLOAT_t fmaxf(TYPE_FLOAT_t x,TYPE_FLOAT_t y)
    TYPE_INT_t isfinite(TYPE_FLOAT_t x)
    enum: INFINITY

@cython.no_gc_clear
cdef class LangevinIntegrator:
    cdef: 
        np.ndarray out_
        TYPE_FLOAT_t dt_
        TYPE_FLOAT_t tmax_
        TYPE_FLOAT_t mod_
        TYPE_INT_t steps_

        np.ndarray drift_
        np.ndarray diffusion_
        np.ndarray rands_

        TYPE_INT_t dpoints_
        np.ndarray dx_
        gsl_interp_accel *acc_drift
        gsl_interp_accel *acc_diff
        gsl_spline *sp_drift
        gsl_spline *sp_diff

    property out:
        def __get__(self):
            return self.out_
        def __set__(self, TYPE_FLOAT_t value):
            raise ValueError("out can not be set manually, use run()")
    
    property dt:
        def __get__(self):
            return self.dt_
        def __set__(self, TYPE_FLOAT_t value):
            raise NotImplementedError("Please instantiate a new LangevinIntegrator instead.")

    property modulus:
        def __get__(self):
            return self.mod_
        def __set__(self, TYPE_FLOAT_t value):
            raise NotImplementedError("Please instantiate a new LangevinIntegrator instead.")

    property tmax:
        def __get__(self):
            return self.tmax_
        def __set__(self, TYPE_FLOAT_t value):
            raise NotImplementedError("Please instantiate a new LangevinIntegrator instead.")

    property drift:
        def __get__(self):
            return self.drift_/self.dt
        def __set__(self, np.ndarray value):
            raise NotImplementedError("Please instantiate a new LangevinIntegrator instead.")

    property diffusion:
        def __get__(self):
            return self.diffusion_**2 / self.dt
        def __set__(self, np.ndarray value):
            raise NotImplementedError("Please instantiate a new LangevinIntegrator instead.")
            

    def __init__(self, drift, diffusion, dt=.001, tmax=1., modulus=2.*np.pi):
        """Langevin Integrator for a given discretization of a drift field A(x)
        and a (possibly position dependent) diffusion coefficient B(x). 

        Both drift field A and diffusion B need to be arrays of the same dimension.
        They are interpolated to give continuous fields which are used for forward integration.
    
        Forward integration is performed with the Euler-Murayama scheme:
        x(t+dt) = x(t) + dt * A(x(t)) + r * sqrt(dt * B(x(t))),
        where r is a normally distributed random number with zero mean and unit variance.
    
        The variable x will be simulated on the continuous interval [0, modulus), with circular boundaries.

        Args:
            drift (float array): Positional drift field [rad/s]
            diffusion (float array): Positional diffusion coefficient [rad^2/s]
            dt (float, optional): Timestep to integrate over [s]
            tmax (float, optional): Maximal time to simulate
            modulus (float, optional): Upper end of the interval, defaults to 2PI
        """
        
        assert dt > 0.0, "dt must be a strictly positive float" 
        self.dt_ = TYPE_FLOAT(dt)

        assert tmax > 0.0, "tmax must be a strictly positive float" 
        self.tmax_ = TYPE_FLOAT(tmax)

        assert modulus > 0.0, "modulus must be strictly positive" 
        self.mod_ = TYPE_FLOAT(modulus)

        # initialize output array
        self.out_ = np.zeros(int(np.ceil(self.tmax / self.dt))).astype(TYPE_FLOAT)
        self.steps_ = self.out_.shape[0]

        # diffusion is internally stored as sqrt(B * dt) for efficiency
        self.diffusion_ = np.sqrt(diffusion.astype(TYPE_FLOAT)*self.dt)
        
        # drift is internally stored as A*dt for efficiency
        self.drift_ = drift.astype(TYPE_FLOAT)*self.dt

        tmp = self.diffusion_.shape[0]

        # dx is used for the GSL interpolator and has tmp+1 elements
        self.dx_ = np.arange(0, self.mod_+self.mod_/tmp, self.mod_/tmp)
        
        # we extend drift and diffusion values on circular boundaries
        self.diffusion_ = np.concatenate((self.diffusion_,[self.diffusion_[0]])).astype(TYPE_FLOAT)
        self.drift_ = np.concatenate((self.drift_,[self.drift_[0]])).astype(TYPE_FLOAT)
        self.dpoints_ = self.diffusion_.shape[0]

        # check that drift, diffusion, and dx have consistent shapes
        assert self.dx_.shape[0] == self.drift_.shape[0]
        assert self.dx_.shape[0] == self.diffusion_.shape[0]
        assert len(self.dx_) == self.dpoints_
        self.c_init()
        
    cdef void c_init(self):
        """Initalizes all c internal objects"""
        self.acc_diff  = gsl_interp_accel_alloc()
        self.acc_drift = gsl_interp_accel_alloc()
        self.sp_diff = <gsl_spline *>gsl_spline_alloc(gsl_interp_cspline_periodic, self.dpoints_)
        self.sp_drift = <gsl_spline *>gsl_spline_alloc(gsl_interp_cspline_periodic, self.dpoints_)

        gsl_spline_init(self.sp_diff, <TYPE_FLOAT_t*>self.dx_.data, <TYPE_FLOAT_t*>self.diffusion_.data, self.dpoints_)
        gsl_spline_init(self.sp_drift, <TYPE_FLOAT_t*>self.dx_.data, <TYPE_FLOAT_t*>self.drift_.data, self.dpoints_)

    @cython.boundscheck(False) 
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void c_run(self) except *:
        """Runs a single forward integraton."""
        cdef unsigned long length, i
        cdef TYPE_FLOAT_t* out_data =  <TYPE_FLOAT_t*>self.out_.data
        cdef TYPE_FLOAT_t* rands_data =  <TYPE_FLOAT_t*>self.rands_.data
        with nogil:
            for i in xrange(1, self.steps_):
                out_data[i] = (out_data[i-1] + gsl_spline_eval(self.sp_drift, out_data[i-1], self.acc_drift) + gsl_spline_eval(self.sp_diff, out_data[i-1], self.acc_diff) * rands_data[i-1] ) % self.mod_
                
                # If cdivision == True above, the c modulus can return negative numbers
                # However, cdivision == False also introduces additional checks for zero division
                # https://github.com/cython/cython/wiki/enhancements-division
                # 
                # We thus want to keep cdivision == True
                # and need to implement a manual modulus for negative numbers
                if out_data[i] < 0:
                    out_data[i] = self.mod_ + out_data[i]

    cpdef run(self, initial, seed=None):
        """Runs a single forward integration (Euler-Murayama) starting from a given initial point
        Attributes:
            initial (float): initial point for the trajectory
            seed (float, optional): seed for the random number generator
        """
        self.out_[0] = TYPE_FLOAT(initial%self.mod_)
        if seed:
            np.random.seed(seed)
        self.rands_ = np.random.randn(self.steps_-1).astype(TYPE_FLOAT)
        self.c_run()

    def __dealloc__(self):
        gsl_spline_free(self.sp_drift);
        gsl_spline_free(self.sp_diff);
        gsl_interp_accel_free(self.acc_drift);
        gsl_interp_accel_free(self.acc_diff);