# coding: utf-8

""" Stellar parameter determination """

__author__ = "Andy Casey <andy@astrowizici.st>"

import logging
import os
from time import time
import cPickle as pickle
from itertools import chain
from random import choice
from string import ascii_letters

import emcee
import numpy as np
import numpy.lib.recfunctions as nprcf
from scipy import optimize as op, ndimage

import atmospheres
import model
import moog
import specutils
import utils
import line

from channel import SpectralChannel

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

solar_abundances = {
    26: 7.50
}

def excitation_ionisation_equilibria(theta, model_atmospheres, equivalent_width_filename, moogsilent,
    line_outliers=False):
    
    no_result = np.array([np.inf]*len(theta))
    teff, xi, logg, m_h = theta

    # Prior:
    if 0 > xi:
        print(theta, "0>xi")
        return no_result

    # Interpolate a model atmopshere.
    atmosphere_filename = os.path.join(moogsilent.twd, "".join([choice(ascii_letters) \
        for _ in xrange(5)]) + ".in")
    while os.path.exists(atmosphere_filename):
        atmosphere_filename = os.path.join(moogsilent.twd, "".join([choice(ascii_letters) \
            for _ in xrange(5)]) + ".in")

    # Calculate abundances from equivalent widths.
    try:
        model_atmospheres.interpolate([teff, logg, m_h], xi, atmosphere_filename)
        abundances = moogsilent.abfind(atmosphere_filename, equivalent_width_filename, parallel=True)
    except:
        print(theta, "fail")
        return no_result

    neutral = (abundances["species"] == 26.)
    ionised = ~neutral
    components = np.array([
        line.Line(abundances["excitation_potential"], abundances["abundance"], outliers=line_outliers).optimise()[0],
        line.Line(np.log10(abundances["equivalent_width"]/abundances["wavelength"]), abundances["abundance"],
            outliers=line_outliers).optimise()[0],
        0.1 * (np.median(abundances[neutral]["abundance"]) - np.median(abundances[ionised]["abundance"])),
        0.1 * (np.median(abundances["abundance"]) - (m_h + solar_abundances[26]))
    ])
    print(theta, np.sum(components**2))
    return components


def ln_likelihood(theta, interpolator, equivalent_widths_filename, moogsilent, fprime=True):

    global count
    print(theta)
    teff, xi, logg, m_h = theta

    # Prior:
    if 0 > xi:
        return [np.inf]*len(theta)

    # Interpolate a model atmopshere.
    atmosphere_filename = os.path.join(moogsilent.twd, "".join([choice(ascii_letters) for _ in xrange(5)]) + ".in")
    while os.path.exists(atmosphere_filename):
        atmosphere_filename = os.path.join(moogsilent.twd, "".join([choice(ascii_letters) for _ in xrange(5)]) + ".in")

    # Calculate abundances from equivalent widths.
    try:
        interpolator.interpolate(atmosphere_filename, teff, logg, m_h, +0.4, xi)
        abundances = moogsilent.abfind(atmosphere_filename, equivalent_widths_filename, parallel=True)
    except:
        return [np.inf]*len(theta)
    
    # Compare "observed" abundances - expected (input) abundance.
    # ASSUME WE ARE ONLY DOING FE
    

    # Calculate the partial derivatives.
    if fprime:
        neutral = (abundances["species"] == 26.)
        ionised = ~neutral
        pre_condition = np.array([
            line.Line(abundances["excitation_potential"], abundances["abundance"], outliers=False).optimise()[0],
            line.Line(np.log10(abundances["equivalent_width"]/abundances["wavelength"]), abundances["abundance"],
                outliers=False).optimise()[0],
            0.1 * (np.mean(abundances[neutral]["abundance"]) - np.mean(abundances[ionised]["abundance"])),
            0.1 * (np.mean(abundances["abundance"]) - (m_h + solar_abundances[26]))
        ])
        print(pre_condition, np.sum(pre_condition**2))

        count += 1.
        return pre_condition/count

    else:
        ivar = 1.0/np.array([0.02]*len(abundances))**2
        ln_like = -0.5 * np.sum((abundances["abundance"] - (m_h + solar_abundances[26]))**2 * ivar - np.log(ivar))
        print(theta, ln_like)

        return ln_like

class Star(object):
    """
    A model star.
    """

    def __init__(self, model_atmosphere_wildmask, transitions, spectra=None, **kwargs):
        """
        Create a model star.

        model_atmosphere_wildmask : wildmask to match model atmosphere files
        transitions : a record array of transitions and relevant synthesis line lists
        spectra : list of spectrum1D objects representing the observed data
        """

        # Sort the spectra from blue to red
        if spectra is None or len(spectra) == 0:
            self.spectra = []
        else:
            self.spectra = [spectra[index] for index in np.argsort([spectrum.dispersion[0] \
                for spectrum in spectra])]

        self.transitions = transitions

        # Create spectral channels for each spectrum, and associate transitions to each.
        self.channels = []
        previous_order_max_disp = 0
        for spectrum in spectra:
            min_disp, max_disp = np.min(spectrum.dispersion), np.max(spectrum.dispersion)
            assert min_disp > previous_order_max_disp, "Spectral channels are assumed not" \
                                                       " to be overlapping."
            # Find transitions that match this.
            indices = (max_disp > transitions["rest_wavelength"]) * (transitions["rest_wavelength"] > min_disp)
            self.channels.append(SpectralChannel(spectrum, transitions[indices], **kwargs))
            previous_order_max_disp = max_disp

        # Create the atmosphere interpolator.
        self.model_atmospheres = atmospheres.Interpolator(model_atmosphere_wildmask)

        # Create a list of parameters.
        self.parameters = [] + self.model_atmospheres.parameters
        return None


    def optimise_equilibria(self, p0=None, method="classical"):
        """
        Perform an excitation and ionisation balance from an initial guess point.
        """

        # [TODO] Get a joint initial esimate based on expected Teff, logg, [M/H] distributions,
        # and known relationships between xi and these parameters.

        p0 = np.array([
            np.random.uniform(3000, 8000),
            np.random.uniform(0, 3),
            np.random.uniform(0, 5),
            np.random.uniform(-5, 1)
        ])
        
        # Initiate a MOOG instance.
        with moog.instance("/tmp",debug=True) as moogsilent:

            # Write equivalent widths to file as these won't change during optimisation.
            with open("ews", "w+") as fp:
                fp.write(moogsilent._format_ew_input(
                    self.transitions[self.transitions["species"].astype(int) == 26]))

            # This is a dirty hack to give fsolve some positive reinforcement.
            _iteration = [0.]
            def wrap_excitation_ionisation_equilibria(theta):
                result = excitation_ionisation_equilibria(theta, self.model_atmospheres, "ews", moogsilent)
                _iteration[0] += 1.
                return result/_iteration[0]

            # The global solver makes use of an empirical Jacobian approximation to quickly move through the
            # parameter space.
            global_solver = op.fsolve(wrap_excitation_ionisation_equilibria, p0, fprime=utils.sp_jacobian,
                col_deriv=1, epsfcn=0, xtol=1e-5, full_output=1)

            # Once the solution is near, the Jacobian is providing too much (false) information to the system,
            # so we perform a scalar optimisation with the squared sum of all components.
            scalar_func = lambda theta: sum(excitation_ionisation_equilibria(theta, self.model_atmospheres, \
                "ews", moogsilent)**2)
            scalar_solver = op.fmin(scalar_func, global_solver[0], xtol=1e-2, ftol=4e-6, full_output=True)

        return scalar_solver



            

"""
Infer the parameters of the model, given the data.

Parameters:
teff            : effective temperature of the atmosphere
logg            : surface gravity of the atmosphere
[M/H]           : mean metallicity of the atmosphere (e.g. this is *not* always [Fe/H])
[alpha/M]       : mean alpha enhancement of the atmosphere over the mean metallicity
xi              : microturbulence in the model atmosphere
{z}_{chan}      : redshift in each observed channel (1 per channel)
{c_i}_{chan}    : continuum coefficients in each observed channel (3-4 per channel)
{sigma}_{chan}  : gaussian smoothing kernel in each observed channel (1 per channel)
log_X_{line}    : abundance of every line of interest (~309 parameters?!)

Total number of free parameters: 5 + 4 + 4*4 + 4 + 309 --> 338 free parameters. Fuck.

Use masks around each line to identify which pixels to use in the comparison. This will
yield us posterior probability distributions in all parameters given the data. I admit,
the problem description looks daunting, but I can't think of any other way to get the
correct PDFs.

p_opt = scalar_solver[0]
ndim = len(p_opt)
sampler = emcee.EnsembleSampler(walkers, ndim, ln_likelihood,
    args=(self.model_atmospheres, "ews", moogsilent), threads=threads)

mean_acceptance_fractions = np.zeros(burn + sample)
initial_pos = np.array([p_opt + 1e-4 * np.random.randn(ndim) for i in range(walkers)])
for i, (pos, lnprob, rstate) in enumerate(sampler.sample(initial_pos, iterations=burn)):
    mean_acceptance_fractions[i] = np.mean(sampler.acceptance_fraction)
    logger.info(u"Sampler has finished step {0:.0f} with <a_f> = {1:.3f}, maximum log probability"\
        " in last step was {2:.3e}".format(i + 1, mean_acceptance_fractions[i],
            np.max(sampler.lnprobability[:, i])))

    if mean_acceptance_fractions[i] in (0, 1):
        raise RuntimeError("mean acceptance fraction is {0:.0f}!".format(mean_acceptance_fractions[i]))

# Reset the chains and sample.
logger.info("Resetting chain...")
chain, lnprobability = sampler.chain, sampler.lnprobability
sampler.reset()

logger.info("Sampling posterior...")
for j, state in enumerate(sampler.sample(pos, iterations=sample)):
    mean_acceptance_fractions[i + j + 1] = np.mean(sampler.acceptance_fraction)

raise a
"""

if __name__ == "__main__":
        
    class spectrum(object):
        pass


    data = np.loadtxt("spectra/uvessun1.txt", skiprows=1)

    blue_channel = spectrum()
    blue_channel.dispersion = data[:,0]
    blue_channel.flux = data[:, 1]
    blue_channel.variance =  np.array([0.001] * len(blue_channel.dispersion))
    blue_channel.inv_var = 1.0/blue_channel.variance


    data = np.loadtxt("spectra/uvessun2.txt", skiprows=1)

    green_channel = spectrum()
    green_channel.dispersion = data[:,0]
    green_channel.flux = data[:, 1]
    green_channel.variance =  np.array([0.001] * len(green_channel.dispersion))
    green_channel.inv_var = 1.0/green_channel.variance

    data = np.loadtxt("spectra/uvessun3.txt", skiprows=1)

    red_channel = spectrum()
    red_channel.dispersion = data[:,0]
    red_channel.flux = data[:, 1]
    red_channel.variance =  np.array([0.001] * len(red_channel.dispersion))
    red_channel.inv_var = 1.0/red_channel.variance

    data = np.loadtxt("spectra/uvessun4.txt", skiprows=1)

    ir_channel = spectrum()
    ir_channel.dispersion = data[:,0]
    ir_channel.flux = data[:, 1]
    ir_channel.variance =  np.array([0.001] * len(ir_channel.dispersion))
    ir_channel.inv_var = 1.0/ir_channel.variance

    #sun = Star([blue_channel], "marcs-sun.model")#, line_lists)
    #sun.optimise_channel(blue_channel, "marcs-sun.model", transitions)

    with open("transitions.pkl", "rb") as fp:
        transitions = pickle.load(fp)

    # Get just blue channel ones
    indices = (5875. > transitions["rest_wavelength"]) * (transitions["rest_wavelength"] > 5645.)

    blue = SpectralChannel(blue_channel, transitions[indices], wl_tolerance=0.1)
    #result = blue.optimise(verbose=True)

    #star = Star("MARCS (2011)", transitions, [blue_channel, green_channel, red_channel, ir_channel], wl_tolerance=0.1)

    # Now we have equivalent widths. Use them to optimise stellar parameters.

    """
    Star(spectral_data, atmospheres, transitions)
    .channels

    (1) Split up transitions to each channel. Warn if same transition goes to multiple channels, or if there is overlap between channels.

    (2) star.optimise(p0)
        Measures EWs in each channel, calculates line trends and optimises them.

    (3) star.infer(p0)
    """


    with open("yong-2012-ngc6752-table2-mg1.pkl", "rb") as fp:
        transitions = pickle.load(fp)
    #yong_star = Star("Castelli & Kurucz (2003)", transitions, [], wl_tolerance=0.1)
    #yong_star = Star("atmospheres/castelli-kurucz/a???at*.dat", transitions, [], wl_tolerance=0.1)
    yong_star = Star("/Users/arc/atmospheres/castelli-kurucz-2004/a???at*.dat", transitions, [], wl_tolerance=0.1)

    #yong_star.infer_sp_from_ew(None)

