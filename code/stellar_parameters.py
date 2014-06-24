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

import atmosphere
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
count = 0.

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

    def __init__(self, model_atmospheres, transitions, spectra=None, **kwargs):
        """
        Create a model star.

        atmospheres : an atmosphere interpolator
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
        if model_atmospheres not in atmosphere.parsers:
            raise KeyError("No matching model atmospheres found. Installed types are: {0}".format(
                ", ".join(atmosphere.parsers.keys())))
        folder, parser = atmosphere.parsers[model_atmospheres]
        self.model_atmospheres = atmosphere.AtmosphereInterpolator(folder, parser())

        # Create a list of parameters.


        self.parameters = ["teff", "logg", "[M/H]", "[alpha/M]", "xi"]
        return None


    def optimise_sp(self, p0=None, elements=("Fe", ), measure=True, line_outliers=True, moog_kwargs=None, **kwargs):
        """
        Calculate equivalent widths for all transitions in all channels, then optimise the stellar
        parameters.
        """

        if p0 is None:

            initial_pos = np.array([
                np.random.uniform(3000, 8000),
                1,
                np.random.uniform(0, 5),
                np.random.uniform(-5, 1)
                ])

            #p0 = [4500, 1.0, 2.0, -1]
        else:
            initial_pos = p0
        if moog_kwargs is None:
            moog_kwargs = {}

        # Get the list of neutral and ionized species for stellar parameter optimisation.
        species = map(utils.element_to_species, elements)
        species += [0.1 + each for each in species]

        if measure:
            # Measure equivalent widths in all channels.
            balance_transitions = []
            for i, channel in enumerate(self.channels):
                if "plot_filename" in kwargs:
                    channel_kwargs = kwargs.copy()
                    channel_kwargs["plot_filename"] = channel_kwargs["plot_filename"].format(i)
                else:
                    channel_kwargs = kwargs.copy()
                xopt, fopt, model_flux = channel.optimise(**channel_kwargs)

                # Identify all the species we need.
                species_indices = np.zeros(len(channel.transitions), dtype=bool)
                for each in species:
                    species_indices += (channel.transitions["species"] == each)
                balance_transitions.append(channel.transitions[species_indices])

            # Use the table of measured equivalent widths to optimise stellar parameters.
            balance_transitions = nprcf.stack_arrays(balance_transitions, usemask=False)

        else:
            balance_transitions = []
            species_indices = np.zeros(len(self.transitions), dtype=bool)
            for each in species:
                species_indices += (self.transitions["species"] == each)
            balance_transitions = self.transitions[species_indices]

        # We'll need an instance of MOOG for this.
        with moog.instance("/tmp", debug=True) as moogsilent:

            # Write transitions and equivalent widths to file.
            ew_filename = os.path.join(moogsilent.twd, "ews")
            with open(ew_filename, "w+") as fp:
                fp.write(moogsilent._format_ew_input(balance_transitions, **moog_kwargs))
            

            fill_value = 1.
            def minimise(theta):

                # Parse theta
                teff, xi, logg, m_h= theta[:4]
                alpha_m = 0.

                if 0 > xi: return np.array([fill_value] * len(theta))

                # Check to see if we have already executed MOOG for these parameters given some
                # tolerance, so that we can avoid unnecessary calls to MOOG.

                # Interpolate a stellar atmosphere and write it to file.
                atmosphere_filename = os.path.join(moogsilent.twd, "model")
                try:
                    thermal_structure = self.model_atmospheres.interpolate_thermal_structure(
                        teff, logg, m_h, alpha_m)
                except:
                    return np.array([fill_value] * len(theta))

                self.model_atmospheres.parser.write_atmosphere(atmosphere_filename,
                    teff, logg, m_h, xi, thermal_structure, clobber=True)

                # Convert equivalent widths to abundances.
                results = moogsilent.abfind(atmosphere_filename, ew_filename, **moog_kwargs)

                # Calculate trend lines, allowing for outliers where appropriate.
                # Trend lines in:
                # - reduced equivalent width vs abundance
                # - excitation potential vs abundance
                # - input model atmosphere - mean Fe abundance
                # - mean Fe abundance - mean Fe II abundance
                excitation_balance = line.Line(x=results["excitation_potential"], y=results["abundance"],
                    yerr=np.array([0.05] * len(results)), outliers=line_outliers).optimise()

                turbulence_balance = line.Line(x=np.log(results["equivalent_width"]/results["wavelength"]),
                    y=results["abundance"], yerr=np.array([0.05] * len(results)), outliers=line_outliers)\
                    .optimise()

                # Find neutral and ionised species.
                int_species = np.array(map(int, results["species"]))
                ionised = results["species"] % int_species > 0
                neutral = ~ionised
                # Match by species. 
                ionisation_balance = 0
                for specie in set(map(int, species)):
                    neutral_species = (int_species == specie) * neutral
                    ionised_species = (int_species == specie) * ionised
                    ionisation_balance += (np.median(results[neutral_species]["abundance"]) \
                        - np.median(results[ionised_species]["abundance"]))/10.

                metallicity_balance = np.median((results[(results["species"] == species[0])]["abundance"] \
                    - (m_h + solar_abundances[int(species[0])]))/10.) # TODO

                components = np.array([excitation_balance[0], turbulence_balance[0], ionisation_balance,
                    metallicity_balance])**2

                print(theta, np.sum(components))
                return components

            def minimise_scalar(theta):
                return np.sum(minimise(theta))
                
            # Do an initial optimisation with fsolve, then one with fmin from the best point?
            vector_opt = op.fsolve(minimise, p0, fprime=utils.sp_jacobian, col_deriv=1, epsfcn=0,
                xtol=1e-10, full_output=1, maxfev=100)

            scalar_opt = op.fmin(minimise_scalar, vector_opt[0])
            raise a
        return result


    def infer_sp_from_ew(self, p0, walkers=50, burn=900, sample=100):
        
        p0 = [4000, 1.06, 1.0, -1]
        #p0 = [np.random.uniform(*boundaries) for boundaries in self.model_atmospheres.boundaries]
         
        p0 = np.array([
            np.random.uniform(3000, 8000),
            np.random.uniform(0, 3),
            np.random.uniform(0, 5),
            np.random.uniform(-5, 1)
        ])

   
        #teff, xi, logg, m_h
        with moog.instance(debug=True) as moogsilent:

            with open("ews", "w+") as fp:
                fp.write(moogsilent._format_ew_input(self.transitions[self.transitions["species"].astype(int) == 26]))

            #scalar_opt = lambda theta: ln_likelihood(theta, self.model_atmospheres, "ews", moogsilent)
            newton = op.fsolve(ln_likelihood, p0, args=(self.model_atmospheres, "ews", moogsilent), fprime=utils.sp_jacobian,
                    col_deriv=1, epsfcn=0, xtol=1e-10, full_output=1)

            raise a
            p0 = op.fmin(scalar_opt, p0)

            ndim = len(p0)
            sampler = emcee.EnsembleSampler(walkers, ndim, ln_likelihood,
                args=(self.model_atmospheres, "ews", moogsilent), threads=24)

            mean_acceptance_fractions = np.zeros(burn + sample)
            initial_pos = np.array([p0 + 1e-4 * np.random.randn(ndim) for i in range(walkers)])
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
"""


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

star = Star("MARCS (2011)", transitions, [blue_channel, green_channel, red_channel, ir_channel], wl_tolerance=0.1)

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
yong_star = Star("Castelli & Kurucz (2003)", transitions, [], wl_tolerance=0.1)
#yong_star.infer_sp_from_ew(None)

