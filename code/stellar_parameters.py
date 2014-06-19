# coding: utf-8

""" Stellar parameter determination """

__author__ = "Andy Casey <andy@astrowizici.st>"

import logging
import os
from time import time
import cPickle as pickle
from itertools import chain

import numpy as np
import numpy.lib.recfunctions as nprcf
from scipy import optimize as op, ndimage

import model
import moog
import specutils
import utils
import line

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Star(object):
    """
    A model star.
    """

    def __init__(self, atmospheres, transitions, spectra, **kwargs):
        """
        Create a model star.

        atmospheres : an atmosphere interpolator
        transitions : a record array of transitions and relevant synthesis line lists
        spectra : list of spectrum1D objects representing the observed data
        """

        # Sort the spectra from blue to red
        self.spectra = [spectra[index] for index in np.argsort([spectrum.dispersion[0] \
            for spectrum in spectra])]

        self.atmospheres = atmospheres
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

        # Create a list of parameters.
        self.parameters = ["teff", "logg", "[M/H]", "xi"]
        return None


    def optimise_sp(self, p0=None, elements=("Fe", ), line_outliers=True, **kwargs):
        """
        Calculate equivalent widths for all transitions in all channels, then optimise the stellar
        parameters.
        """

        if p0 is None:
            raise NotImplementedError("need to test limits with an atmosphere interpolator")

        # Get the list of neutral and ionized species for stellar parameter optimisation.
        species = map(utils.element_to_species, elements)
        species += [0.1 + each for each in species]

        # Measure equivalent widths in all channels.
        balance_transitions = []
        for i, channel in enumerate(self.channels):
            xopt, fopt, model_flux = channel.optimise(**kwargs)

            # Identify all the species we need.
            species_indices = np.zeros(len(channel.transitions), dtype=bool)
            for each in species:
                species_indices += (channel.transitions["species"] == each)
            balance_transitions.append(channel.transitions[species_indices])

        balance_transitions = nprcf.merge_arrays(*balance_transitions)

        # Use the table of measured equivalent widths to optimise stellar parameters.
        # We'll need an instance of MOOG for this.

        with moog.instance("/tmp", debug=True) as moogsilent:

            # Write transitions and equivalent widths to file.
            ew_filename = os.path.join(moogsilent.twd, "ews")
            with open(ew_filename, "w+") as fp:
                fp.write(moogsilent._format_ew_input(balance_transitions, **kwargs))
            
            abundances = moogsilent.abfind("marcs-sun.model", ew_filename, **kwargs)
            plt.scatter(np.log(abundances["equivalent_width"]/abundances["wavelength"]), abundances["abundance"])
            raise a
            
            def minimise(theta):

                # Check to see if we have already executed MOOG for these parameters given some
                # tolerance, so that we can avoid unnecessary calls to MOOG.

                # Interpolate a stellar atmosphere and write it to file.

                # Convert equivalent widths to abundances.
                results = moogsilent.abfind(atmosphere_filename, ew_filename, **kwargs)

                # Calculate trend lines, allowing for outliers.
                # Trend lines in:
                # - reduced equivalent width vs abundance
                # - excitation potential vs abundance
                # - input model atmosphere - mean Fe abundance?
                # - mean Fe abundance - mean Fe II abundance?
                excitation_balance = line.Line(x=results["excitation_potential"], y=results["abundance"],
                    outliers=line_outliers, yerr=[0.05] * len(results), ).optimise()

                turbulence_balance = line.Line(x=np.log(results["equivalent_width"]/results["wavelength"]),
                    y=results["abundance"], yerr=[0.05] * len(results), outliers=line_outliers).optimise()

                # Find neutral and ionised speces
                int_species = map(int, results["species"])
                ionised = results["species"] % int_species > 0
                neutral = not ionised
                # Match by species. 
                ionisation_balance = 0           
                for specie in set(map(int, species)):
                    neutral_species = (int_species == specie) * neutral
                    ionised_species = (int_species == specie) * ionised
                    ionisation_balance += (np.mean(neutral_species["abundance"]) \
                        - np.mean(ionised_species["abundance"]))**2


                # Return a sum of slopes.

                None

            # Run minimise for p0.

            # scipy optimise

        return None


    def infer(self, ):
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

        raise NotImplementedError


class SpectralChannel(object):

    def __init__(self, data, transitions, redshift=False, continuum_order=-1, outliers=False,
        wl_tolerance=0, wl_cont=1):

        self.dispersion = data.dispersion
        self.data = data.flux
        self._finite = np.isfinite(self.data)
        self.variance = data.variance

        self.transitions = transitions

        # Does the transitions table have an equivalent width field? If not, add one.
        if "equivalent_width" not in self.transitions.dtype.names:
            self.transitions = nprcf.append_fields(self.transitions, "equivalent_width",
                [np.nan] * len(self.transitions), usemask=False)

        self.wl_cont = wl_cont
        self.redshift = redshift
        self.outliers = outliers
        self.wl_tolerance = wl_tolerance
        self.continuum_order = continuum_order

        # Create parameter list.
        self.parameters = []
        if redshift: self.parameters.append("z")
        self.parameters.append("smoothing_sigma")
        self.parameters.extend(["c_{0}".format(i) for i in range(continuum_order + 1)])

        # Outliers?
        if outliers:
            self.parameters.extend(["Po", "Vo", "Yo"])
            
        # Line depths.
        self.parameters.extend(["ld_{0}".format(i) for i in range(len(transitions))])

        # Are wavelengths allowed to vary?
        if wl_tolerance > 0:
            self.parameters.extend(["wl_{0}".format(i) for i in range(len(transitions))])

        # Check for nearby-ish lines.
        sorted_wavelengths = np.sort(self.transitions["rest_wavelength"])
        distances = np.diff(sorted_wavelengths)
        if wl_cont > 0 and wl_cont > np.min(distances):
            nearby_line_indices = np.where(wl_cont > distances)[0]
            for nearby_index in nearby_line_indices:
                logging.warn("Transitions at {0:.2f} and {1:.2f} are very close ({2:.2f} < {3:.2f} Angstroms)".format(
                    sorted_wavelengths[nearby_index], sorted_wavelengths[nearby_index + 1], 
                    float(np.abs(np.diff(sorted_wavelengths[nearby_index:nearby_index+2]))), wl_cont))


    def _fit(self, theta, full_output, verbose=False):

        # Create continuum shape.
        if self.continuum_order > -1:
            continuum_coefficients = [theta[self.parameters.index("c_{0}".format(i))] \
                for i in range(self.continuum_order + 1)]
            continuum = np.polyval(continuum_coefficients, self.dispersion)
        
        else:
            continuum = np.ones(len(self.data))
        
        if self.outliers:
            Po, Yo, Vo = [theta[self.parameters.index(each)] for each in ("Po", "Yo", "Vo")]
            if not (1 > Po > 0) or 0 > Vo:
                return np.inf

        # Add absorption lines.
        for i, transition in enumerate(self.transitions):
            if self.wl_tolerance > 0:
                wavelength = theta[self.parameters.index("wl_{0}".format(i))]

            else:
                wavelength = transition["rest_wavelength"]

            sigma = theta[0]
            depth = theta[self.parameters.index("ld_{0}".format(i))]
            if not (1 >= depth >= 0) or 0 > sigma \
            or abs(wavelength - transition["rest_wavelength"]) > self.wl_tolerance:
                return np.inf

            z = args[0] if self.redshift else 0
            wavelength *= (1. + z)

            if self.wl_cont > 0:
                indices = self.dispersion.searchsorted([
                    wavelength - self.wl_cont,
                    wavelength + self.wl_cont
                ])
                x = self.dispersion.__getslice__(*indices)
                y = continuum.__getslice__(*indices)

                continuum.__setslice__(indices[0], indices[1],
                    y * self._absorption_line_(wavelength, depth, sigma, x=x)
                )

            else:
                continuum *= self._absorption_line_(wavelength, depth, sigma)

        ivar = 1.0/self.variance
        chi_sq = (continuum - self.data)**2 * ivar
        
        if self.outliers:

            signal_ln_like = -0.5 * (chi_sq - np.log(ivar))

            ivar = 1.0/(self.variance + Vo)
            outlier_ln_like = -0.5 * ((continuum - Yo)**2 * ivar - np.log(ivar))

            # Make this the negative log-likelihood, since this function will only be used in optimisation.
            ln_like = -np.sum(np.logaddexp(np.log(1-Po) + signal_ln_like, np.log(Po) + outlier_ln_like))

            if verbose:
                print(ln_like)
            
            if full_output:
                return (ln_like, continuum)
            return ln_like

        # No outlier modelling; just return chi-sq.
        chi_sq = np.sum(chi_sq)
        if verbose:
            print(chi_sq)
        if full_output:
            return (chi_sq, continuum)
        return chi_sq


    def optimise(self, force=False, verbose=False, line_kwargs=None, channel_kwargs=None):
        # Optimise the channel and line parameters.

        op_line_kwargs = { "xtol": 1e-8, "maxfun": 10e3, "maxiter": 10e3,
            "full_output": True, "disp": False }
        op_channel_kwargs = { "ftol": 1e-3, "maxfun": 10e4, "maxiter": 10e4,
            "full_output": True, "disp": False } 

        if line_kwargs is not None:
            op_line_kwargs.update(line_kwargs)
        if channel_kwargs is not None:
            op_channel_kwargs.update(channel_kwargs)
        
        # Let's do a first pass for each line, then we'll iterate on the channel globally.
        default_p0 = {
            "z": 0,
            "smoothing_sigma": 0.1,
            "Po": 0.5,
            "Yo": np.median(self.data[self._finite]),
            "Vo": np.std(self.data[self._finite])**2
        }

        # Wavelengths
        default_p0.update(dict(zip(
            ["wl_{0}".format(i) for i in range(len(self.transitions))],
            [each["rest_wavelength"] for each in self.transitions]
        )))

        # Continuum coefficients
        if self.continuum_order >= 0:
            continuum_coefficients = np.polyfit(self.dispersion, self.data, self.continuum_order)
            default_p0.update(dict(zip(
                ["c_{0}".format(i) for i in range(self.continuum_order + 1)],
                continuum_coefficients
            )))

        continuum_shape = np.ones(len(self.data))
        if self.continuum_order > -1:
            continuum_shape *= np.polyval(continuum_coefficients, self.dispersion)

        xopts = []
        # Fit each line using the continuum we've set.
        for i, transition in enumerate(self.transitions):

            def fit_absorption_line(args, continuum, full_output):
                
                continuum = continuum.copy()

                if self.wl_tolerance > 0:
                    depth, sigma, wavelength = args

                else:
                    depth, sigma = args
                    wavelength = transition["rest_wavelength"]

                if not (1 > depth > 0) or 0 > sigma or abs(wavelength - transition["rest_wavelength"]) > self.wl_tolerance:
                    return np.inf

                z = args[0] if self.redshift else 0
                wavelength *= (1. + z)

                if self.wl_cont > 0:
                    indices = self.dispersion.searchsorted([
                        wavelength - self.wl_cont,
                        wavelength + self.wl_cont
                    ])
                    x = self.dispersion.__getslice__(*indices)
                    y = continuum.__getslice__(*indices)

                    continuum.__setslice__(indices[0], indices[1],
                        y * self._absorption_line_(wavelength, depth, sigma, x=x)
                    )

                else:
                    continuum *= self._absorption_line_(wavelength, depth, sigma)

                chi_sq = np.sum((continuum - self.data)**2/self.variance)
                if full_output:
                    return (chi_sq, continuum)
                return chi_sq

            index = self.dispersion.searchsorted(transition["rest_wavelength"])

            line_p0 = [1. - self.data[index]/continuum_shape[index], default_p0["smoothing_sigma"]]
            if self.wl_tolerance > 0:
                line_p0.append(transition["rest_wavelength"])

            xopt, fopt, niter, nfuncs, warnflag = op.fmin(fit_absorption_line, line_p0, args=(continuum_shape, False),
                **op_line_kwargs)
            xopts.append(xopt)
            if warnflag > 0:
                message = [
                    "Che problem?",
                    "Maximum number of function evaluations made",
                    "Maximum number of iterations made"
                ]
                logging.warn("{0} for transition at {1}".format(message[warnflag], transition["rest_wavelength"])) 

            # Update the default_p0 values.
            if self.wl_tolerance > 0:
                default_p0["wl_{0}".format(i)] = xopt[2]
            default_p0["ld_{0}".format(i)] = np.clip(xopt[0], 0, 1)

        default_p0["smoothing_sigma"] = abs(np.median([xopt[1] for xopt in xopts]))
        
        """
        # Produce a plot.
        continuum = continuum_shape.copy()
        for transition, result in zip(self.transitions, line_fits):
            if self.wl_tolerance > 0:
                depth, sigma, wavelength = result["x"]
            else:
                depth, sigma = result["x"]
                wavelength = transition["rest_wavelength"]

            continuum *= self._absorption_line_(wavelength, depth, sigma)

        plt.plot(self.dispersion, self.data, 'k')
        plt.plot(self.dispersion, continuum, 'b')
        raise a
        """

        # Optimise the parameters globally if required.
        # Note: the smoothing kernel is always a global parameter, but the median of the line sigmas
        #       is a sufficient approximation. So here we are just checking for either redshift or
        #       continuum coefficients.
        channel_p0 = np.array([default_p0[parameter] for parameter in self.parameters])
        if self.continuum_order > -1 or "z" in self.parameters or force:
            xopt, fopt, niter, nfuncs, warnflag = op.fmin(self._fit, channel_p0, args=(False, verbose),
                **op_channel_kwargs)
            if warnflag > 0:
                message = [
                    "Che problem?",
                    "Maximum number of function evaluations made",
                    "Maximum number of iterations made"
                ]
                logging.warn("{0} for channel fit. Optimised values may be (even more) inaccurate.".format(
                    message[warnflag]))

        else:
            xopt = channel_p0

        """
        fig, ax = plt.subplots()
        ax.plot(self.dispersion, self.data,'k')
        try:
            ax.plot(self.dispersion, self._fit(channel_p0, True)[1], 'r')
        except:
            None
        ax.plot(self.dispersion, self._fit(xopt, True)[1], 'b')
        raise a
        """

        fopt, model = self._fit(xopt, True)

        # Calculate equivalent widths in the same units as the dispersion.
        logging.warn("Assuming spectral dispersion units are Angstroms. Measured equivalent widths stored as mA.")
        self.transitions["equivalent_width"] = np.array([xopt[self.parameters.index("ld_{0}".format(i))] \
            for i in range(len(self.transitions))]) * xopt[0] * 1000. * 2.65 # TODO: Check integral.
        return (xopt, fopt, model)


    def _absorption_line_(self, wavelength, depth, sigma, x=None):
        if x is None:
            x = self.dispersion
        return 1. - depth * np.exp(-(x - wavelength)**2 / (2 * sigma**2))







data = np.loadtxt("spectra/uvessun2.txt", skiprows=1)

class spectrum(object):
    pass

blue_channel = spectrum()
blue_channel.dispersion = data[:,0]
blue_channel.flux = data[:, 1]
blue_channel.variance =  np.array([0.001] * len(blue_channel.dispersion))
blue_channel.inv_var = 1.0/blue_channel.variance

#sun = Star([blue_channel], "marcs-sun.model")#, line_lists)
#sun.optimise_channel(blue_channel, "marcs-sun.model", transitions)

with open("transitions.pkl", "rb") as fp:
    transitions = pickle.load(fp)

# Get just blue channel ones
indices = (5875. > transitions["rest_wavelength"]) * (transitions["rest_wavelength"] > 5645.)

blue = SpectralChannel(blue_channel, transitions[indices], wl_tolerance=0.1)
#result = blue.optimise(verbose=True)

star = Star(None, transitions, [blue_channel], wl_tolerance=0.1)

# Now we have equivalent widths. Use them to optimise stellar parameters.

"""
Star(spectral_data, atmospheres, transitions)
.channels

(1) Split up transitions to each channel. Warn if same transition goes to multiple channels, or if there is overlap between channels.

(2) star.optimise(p0)
    Measures EWs in each channel, calculates line trends and optimises them.

(3) star.infer(p0)
"""

