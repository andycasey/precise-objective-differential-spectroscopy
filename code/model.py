# coding: utf-8

""" Spectral fitting """

__author__ = "Andy Casey <andy@astrowizici.st>"

import logging
from itertools import chain

import numpy as np
import scipy.optimize as op, scipy.ndimage as ndimage, scipy.interpolate as interpolate

import moog
import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ApproximateModelSpectrum(object):

    def __init__(self, line_list_filename, model_atmosphere_filename, elements,
        convolve=False, v_rad=False, continuum=False, continuum_order=-1, outliers=False,
        moog_instance=None):
        """

        Convolve and v_rad can be True/False or actual values. If True, it's a
        free parameter. Otherwise fixed.

        Continuum can either be True (in which case continuum_order should be
        specified), or a lambda function that takes dispersion as a single value
        (in which case the continuum enters multiplicatively)

        """

        if isinstance(elements, (str, unicode)):
            elements = [elements]

        # Check the elements
        assert len(elements) > 0
        assert map(utils.element_to_species, elements)

        self.parameters = []
        self.fixed_parameters = {}

        self.moog_kwargs = {
            "wl_cont": 2.,
            "wl_edge": 2.,
            "parallel": True
        }
    
        self.elements = elements
        self.line_list_filename = line_list_filename
        self.model_atmosphere_filename = model_atmosphere_filename
        self.moog_instance = None

        # Determine the parameters
        if convolve is True:
            self.parameters.append("convolve")
        elif convolve is not False and isinstance(convolve, (float, int)) and convolve > 0:
            # Convolution is fixed.
            self.fixed_parameters["convolve"] = convolve

        if v_rad is True:
            self.parameters.append("v_rad")
        elif v_rad is not False and isinstance(v_rad, (float, int)):
            # Velocity is fixed.
            self.fixed_parameters["v_rad"] = v_rad

        self.continuum_order = -1
        if continuum is True:
            self.continuum_order = continuum_order
            if continuum_order >= 0:
                self.parameters.extend(["c_{0}".format(i) for i in range(continuum_order + 1)])
                
            else:
                logging.warn("Continuum was specified as True but 0 > continuum_order")

        elif hasattr(continuum, "__call__"):
            self.fixed_parameters["continuum"] = continuum

        if outliers:
            self.parameters.extend(["Pb", "Vb", "Yb"])

        self.parameters.extend(map(lambda x: "Z_{0}".format(x), self.elements))

        # Determine wl_min and wl_max
        try:
            wavelengths = np.loadtxt(line_list_filename, usecols=(0, ))
        except (ValueError, TypeError) as e:
            # First row is probably a comment
            wavelengths = np.loadtxt(line_list_filename, usecols=(0, ), skiprows=1)

        self.moog_kwargs.update({
            "wl_min": np.min(wavelengths) - self.moog_kwargs["wl_edge"],
            "wl_max": np.max(wavelengths) + self.moog_kwargs["wl_edge"],
        })

        # Ensure we haven't done anything silly
        assert len(set(self.parameters).intersection(self.fixed_parameters.keys())) == 0
        assert len(set(self.parameters)) == len(self.parameters)


    def __call__(self, theta):
        """
        Return model dispersion and flux.
        """

        theta_dict = dict(zip(self.parameters, theta))
        get_parameter = lambda p, d: theta_dict.get(p, self.fixed_parameters.get(p, d)) 

        # Do abundances
        abundances = {}
        for element in self.elements:
            species = np.floor(utils.element_to_species(element))
            abundances[species] = [theta_dict["Z_"+element]]

        # Synthesise flux
        if self.moog_instance is not None:
            model_dispersion, model_fluxes = self.moog_instance.synth(self.model_atmosphere_filename,
                self.line_list_filename, abundances=abundances, **self.moog_kwargs)

        else:
            with moog.instance() as moogsilent:
                model_dispersion, model_fluxes = moogsilent.synth(self.model_atmosphere_filename,
                    self.line_list_filename, abundances=abundances, **self.moog_kwargs)

        # Select only the first spectrum
        model_flux = model_fluxes[0]

        # Any convolution to apply?
        convolve = abs(get_parameter("convolve", 0))
        if convolve > 0:
            ndimage.gaussian_filter(model_flux, convolve, output=model_flux)

        # Any radial velocity to apply?
        # TODO: YOU SHOULD DO THIS ON A DISPERSION THAT IS UNIFORM IN LOG-LAMBDA
        z = get_parameter("v_rad", 0)/299792458e-3
        model_dispersion *= 1. + z

        # Any continuum transformation to apply?
        if "continuum" in self.fixed_parameters:
            # Continuum here is a function that takes dispersion as a single input
            continuum = self.fixed_parameters["continuum"](model_dispersion)
            model_flux *= continuum

        elif self.continuum_order >= 0:
            continuum = np.polyval([theta_dict["c_{0}".format(i)] \
                for i in range(self.continuum_order + 1)], model_dispersion)
            model_flux *= continuum

        return (model_dispersion, model_flux)


    def _mask(self, observed_dispersion, rest_frame_regions, z, mask_value=np.nan):

        mask = np.ones(len(observed_dispersion))
        for region in rest_frame_regions:
            shifted_region = region * (1. + z)
            indices = observed_dispersion.searchsorted(shifted_region)
            mask.__setslice__(indices[0], indices[1] + 1, mask_value)

        return mask


    def optimise(self, data, p0=None, p0_abundance_range=[-1, 0, 1], masks=None, xtol=0.01,
        maxiter=10, full_output=False, op_kwargs=None):

        print("Fitting as {0}, {1}".format(self.parameters, self.fixed_parameters))
        dispersion, flux, variance = data.dispersion, data.flux, data.variance

        # Slice the data, clean up non-finite fluxes, etc
        indices = np.searchsorted(data.dispersion,
            [self.moog_kwargs["wl_min"], self.moog_kwargs["wl_max"]])
        dispersion = data.dispersion.__getslice__(*indices)
        flux = data.flux.__getslice__(*indices)
        variance = data.variance.__getslice__(*indices)
        ivar = 1.0/variance

        finite = np.isfinite(flux)

        # Approximate the wavelength sampling for MOOG
        self.moog_kwargs["wl_step"] = np.median(np.diff(dispersion))

        if masks is None:
            masks = []

        if p0 is None:
            p0 = {"c_0": 1., "Pb": 0.5, "Vb": 1., "Yb": np.mean(flux)}
        p0 = [p0.get(parameter, 0) for parameter in self.parameters]
        p0_abundance_range = np.array(p0_abundance_range)

        if op_kwargs is None:
            op_kwargs = {"disp": False, "full_output": True}
        else:
            if "xtol" in op_kwargs:
                del op_kwargs["xtol"]
            op_kwargs["full_output"] = True

        # Let's synthesize 5 spectra at once then use them for interpolation
        abundances = {}
        for element in self.elements:
            species = np.floor(utils.element_to_species(element))
            abundances[species] = p0_abundance_range
            
        niter, warnflag, converged, previous_abundances = 0, 0, False, None
        while not converged:
            
            if self.moog_instance is not None:
                model_dispersion, model_fluxes = self.moog_instance.synth(self.model_atmosphere_filename,
                    self.line_list_filename, abundances=abundances, **self.moog_kwargs)
            else:
                with moog.instance() as moogsilent:
                    model_dispersion, model_fluxes = moogsilent.synth(self.model_atmosphere_filename,
                        self.line_list_filename, abundances=abundances, **self.moog_kwargs)

            # Create an interpolator to interpolate between these spectra
            if len(self.elements) == 1:
                interpolator = interpolate.interp1d(abundances.values()[0], np.array(model_fluxes).T,
                    bounds_error=False)

            else:
                raise NotImplementedError("use LinearNDInterpolator?")

            def ln_prior(theta):

                theta_dict = dict(zip(self.parameters, theta))

                Pb = theta_dict.get("Pb", 0)
                if Pb > 1 or 0 > Pb:
                    return -np.inf

                Vb = theta_dict.get("Vb", 1)
                if 0 >= Vb:
                    return -np.inf

                return 0

            # Define our scalar function to fit the model to the data
            def ln_likelihood(theta):

                theta_dict = dict(zip(self.parameters, theta))
                get_parameter = lambda p, d: theta_dict.get(p, self.fixed_parameters.get(p, d)) 

                interpolated_flux = interpolator(*[theta_dict["Z_"+element] for element in self.elements])

                # Any convolution to apply?
                convolve = abs(get_parameter("convolve", 0))
                if convolve > 0:
                    ndimage.gaussian_filter(interpolated_flux, convolve, output=interpolated_flux)

                # Any radial velocity to apply?
                z = get_parameter("v_rad", 0)/299792458.0e-3

                # Must match model dispersion onto observed dispersion
                # TODO: YOU SHOULD DO THIS ON A DISPERSION THAT IS UNIFORM IN LOG-LAMBDA
                interpolated_flux = np.interp(dispersion, model_dispersion * (1. + z),
                    interpolated_flux, left=np.nan, right=np.nan)

                # Any continuum transformation to apply?
                if "continuum" in self.fixed_parameters:
                    # Continuum here is a function that takes dispersion as a single input
                    continuum = self.fixed_parameters["continuum"](dispersion)
                    interpolated_flux *= continuum

                elif self.continuum_order >= 0:
                    continuum = np.polyval([theta_dict["c_{0}".format(i)] \
                        for i in range(self.continuum_order + 1)], dispersion)
                    interpolated_flux *= continuum

                # Any masks to apply?
                interpolated_flux *= self._mask(dispersion, masks, z)

                photosphere_likelihood = -0.5 * ((flux - interpolated_flux)**2 * ivar - np.log(ivar))

                if "Pb" in theta_dict:
                    outlier_ivar = 1.0/(variance + theta_dict["Vb"])
                    outlier_likelihood = -0.5 * ((flux - theta_dict["Yb"])**2 * outlier_ivar - np.log(outlier_ivar))

                    Pb = theta_dict["Pb"]
                    likelihood = np.logaddexp(np.log(1. - Pb) + photosphere_likelihood, np.log(Pb) + outlier_likelihood)

                else:
                    likelihood = photosphere_likelihood

                finite = np.isfinite(likelihood)
                if not np.any(finite):
                    return -np.inf

                return np.sum(likelihood[finite])

            def ln_prob(theta):
                return ln_prior(theta) + ln_likelihood(theta)

            # Optimise!
            opt_p0, f_opt_p0, direc, op_iter, op_funcalls, op_warnflag = op.fmin_powell(
                lambda theta: -ln_prob(theta), p0, xtol=xtol, **op_kwargs)
            if len(self.parameters) == 1:
                opt_p0 = [opt_p0]
            opt_p0_dict = dict(zip(self.parameters, opt_p0))

            # Check for convergence
            current_abundances = np.array([opt_p0_dict["Z_"+element] for element in self.elements])
            if previous_abundances is not None:
                if np.all(xtol >= np.abs(current_abundances - previous_abundances)):
                    # Achievement unlocked: convergence.
                    break

            # No convergence. Prepare for the next loop.
            previous_abundances = current_abundances

            # Half the p0_range and check for convergence
            for element in self.elements:
                species = np.floor(utils.element_to_species(element))

                opt_value = opt_p0_dict["Z_"+element]

                n_spectra = len(abundances[species])
                half_range = np.ptp(abundances[species])/2.
                new_abundance_range = np.linspace(opt_value - half_range/2., opt_value + half_range/2., n_spectra)
                abundances[species] = new_abundance_range
                
            p0 = opt_p0

            niter += 1
            if niter >= maxiter:
                warnflag = 1
                logging.warn("Maximum number of iterations reached.")
                break

        if full_output:
            return opt_p0, f_opt_p0, warnflag

        return opt_p0


"""
line_lists = \
['linelists/lin56466lab',
 'linelists/lin56514lab',
 'linelists/lin56523lab',
 'linelists/lin56613lab',
 'linelists/lin56790lab',
 'linelists/lin56802lab',
 'linelists/lin56894lab',
 'linelists/lin56960lab',
 'linelists/lin57047lab',
 'linelists/lin57054lab',
 'linelists/lin57164lab',
 'linelists/lin57204lab',
 'linelists/lin57208lab',
 'linelists/lin57244lab',
 'linelists/lin57317lab',
 'linelists/lin57322lab',
 'linelists/lin57394lab',
 'linelists/lin57418lab',
 'linelists/lin57520lab',
 'linelists/lin57663lab',
 'linelists/lin57750lab',
 'linelists/lin57784lab',
 'linelists/lin58067lab',
 'linelists/lin58092lab',
 'linelists/lin58100lab',
 'linelists/lin58218lab',
 'linelists/lin58236lab',
 'linelists/lin58480lab',
 'linelists/lin58481lab',
 'linelists/lin58496lab',
 'linelists/lin58531lab',
 'linelists/lin58550lab',
 'linelists/lin58587lab',
 'linelists/lin58595lab',
 'linelists/lin58611lab',
 'linelists/lin58623lab',
 'linelists/lin58664lab']

models = []
for line_list in line_lists:
    models.append(ApproximateModelSpectrum(line_list, "marcs-sun.model", "Fe", v_rad=True,
    convolve=True, continuum=True, outliers=True, continuum_order=0))

data = np.loadtxt("spectra/uvessun2.txt", skiprows=1)

class spectrum(object):
    pass

b = spectrum()

b.dispersion = data[:,0]
b.flux = data[:,1]
b.variance =  np.array([0.000001] * len(data))

# Fit it

"""
