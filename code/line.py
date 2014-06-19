# coding: utf-8

""" Fit a line to data. """

import numpy as np
import scipy.stats as statistics, scipy.optimize as op

__all__ = ["Line"]

class Line(object):
    """ A straight line model. """
    
    def __init__(self, x, y, yerr, outliers=True):

        assert len(x) == len(y)
        assert len(yerr) == len(y)

        self.x = x
        self.y = y
        self.yerr = yerr
        
        self.parameters = ["m", "b"]
        self.outliers = outliers
        if outliers:
            self.parameters.extend(["Po", "Yo", "Vo"])
        return None


    def optimise(self, xtol=1e-4, ftol=1e-8, maxfun=10e3, maxiter=10e3, disp=False, **kwargs):
        """
        Optimise the model parameters given the data.
        """

        slope, intercept, r_value, p_value, stderr = statistics.linregress(self.x, self.y)

        p0 = [slope, intercept]
        if self.outliers:
            p0.extend([0.5, np.median(self.y), np.std(self.y)])

        # Create the likelihood function.
        def ln_like(theta):

            m, b = theta[:2]
            line = m * self.x + b

            ivar = 1.0/self.yerr**2
            line_likelihood = -0.5 * ((line - self.y)**2 * ivar - np.log(ivar))
            if not self.outliers:
                return np.sum(line_likelihood)

            Po, Yo, Vo = theta[2:]
            # A bit of prior information here:
            if not (1 > Po > 0) or 0 > Vo:
                return -np.inf

            outlier_ivar = 1.0/(self.yerr**2 + Vo)
            outlier_likelihood = -0.5 * ((Yo - self.y)**2 * outlier_ivar - np.log(outlier_ivar))
            likelihood = np.sum(np.logaddexp(np.log(1 - Po) + line_likelihood, np.log(Po) + outlier_likelihood))
            return likelihood

        # Optimise.
        minimise = lambda theta: -ln_like(theta)
        return op.fmin(minimise, p0, xtol=xtol, ftol=ftol, maxfun=maxfun, maxiter=maxiter, disp=False, **kwargs)

if __name__ == "__main__":

    # DEBUG MODE
    x, y = np.loadtxt("data", unpack=True)
    yerr = np.array([0.05] * len(y))
    line_no_outliers = Line(x, y, yerr=yerr, outliers=False)
    m1, b1 = line_no_outliers.optimise()

    line_with_outliers = Line(x, y, yerr=yerr, outliers=True)
    m2, b2, po, yo, vo = line_with_outliers.optimise()

    plt.errorbar(x, y, yerr=yerr, fmt=None, c='k')
    plt.scatter(x,y, facecolor='k')
    plt.plot(x, m1 * x + b1, 'r')
    plt.plot(x, m2 * x + b2, 'b')


