# coding: utf-8

""" Stellar parameter determination """

__author__ = "Andy Casey <andy@astrowizici.st>"

import logging
from itertools import chain

import numpy as np

import model
import specutils
import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Star(object):
    """
    A stellar object.
    """

    def __init__(self, data, model_atmospheres, line_lists, continuum_orders=2, **kwargs):
        """
        Initialise the object.
        """

        if not isinstance(data, (list, tuple)):
            data = [data]

        for spectrum in data:
            if not isinstance(spectrum, specutils.Spectrum1D):
                raise TypeError("Expected data to be a list-type of Spectrum1D objects.")

        # Continuum orders can be the same length as data, or an integer-like quantity
        if not isinstance(continuum_orders, (int, float)):
            if len(continuum_orders) != len(data):
                raise ValueError("Expected {0} continuum order quantities but got {1}".format(
                    len(data), len(continuum_orders)))

            continuum_orders = map(int, continuum_orders)

        else:
            continuum_orders = [int(continuum_orders)] * len(data)

        if not isinstance(line_lists, (list, tuple)):
            line_lists = [line_lists]

        self.data = data
        self.model_atmospheres = model_atmospheres
        self.line_lists = line_lists

        # Determine the spectral coverage of the data, and associate
        # line lists to each channel
        self._channel_associations = [range(len(self.line_lists))]

        if len(data) > 1:
            raise NotImplementedError("haven't worked out associations yet")

        # Determine parameters
        self.parameters = ["Teff", "logg", "[M/H]", "[alpha/M]", "xi"]

        # Separate redshifts per channel
        self.parameters.extend(["z_{0}".format(i) for i in range(data)])

        # Smoothing in each channel
        self.parameters.extend(["convolve_{0}".format(i) for i in range(data)])

        # Continuum parameters
        for channel, continuum_order in enumerate(continuum_orders):
            self.parameters.extend(["c_{0}_{1}".format(channel, j) for j in range(continuum_order)])
        return None


    def optimise_stellar_parameters(self, p0=None):
        """
        Optimise the stellar parameters (Teff, logg, etc) for the given source.
        """

        # Determine some strong initial guess points for all parameters

        # For a stellar parameter, determine the continuum, line abundances, etc

        # Calculate slopes, including allowing for lines as outliers

        # Minimise those slopes using the Jacobian approximation and get the stellar parameters


        raise NotImplementedError


    def optimise_atomic_line_abundances(self, teff, logg, m_h, alpha_fe, xi, p0=None):
        """
        For a given model atmosphere, what are all the line abundances?
        """

        # Parse theta:
        # stellar parameters, continuum_parameters, radial_velocity, smoothing

        #if p0 is None:
        #    p0 = {}
        #p0 = [p0.get(parameter, 0) for parameter in self.parameters[len(stellar_parameters):]]

        with moog.instance(chars="20") as moogsilent:

            # Generate stellar atmosphere
            thermal_structure = self.model_atmospheres.interpolate_thermal_structure(
                teff, logg, m_h, alpha_fe)

            atmosphere_filename = os.path.join(moogsilent.twd, "atmosphere")
            self.model_atmospheres.parser.write_atmosphere(
                atmosphere_filename, teff, logg, m_h, xi, thermal_structure)

            # We seek to determine channel parameters (smoothing, radial velocity and continuum
            # coefficients) at the same time as determining individual line abundances.

            # So we need a function that will model determine the parameters for each channel
            all_line_abundances = []
            channel_parameters = []
            for channel, (spectrum, line_list_filenames) in enumerate(zip(self.data, self._channel_associations)):

                # This line can be parallelised -- do one channel per core.
                opt_channel_parameters, line_abundances = self._optimise_channel_parameters(
                    atmosphere_filename, spectrum, line_list_filenames, moogsilent)

                all_line_abundances.extend(line_abundances)

        return all_line_abundances


    def _optimise_channel_parameters(self, model_atmosphere_filename, spectrum, line_list_filenames, moog_instance,
        v_rad=True, convolve=True, continuum=True, continuum_order=2):
        """
        Determine the parameters for this channel, and all the lines we have in it.
        """

        # These arguments won't change during this level of optimisation.
        model_kwargs = {
            "model_atmosphere_filename": model_atmosphere_filename,
            "full_output": True,
            "moog_instance": moog_instance,
            "element": ["Fe"] # TODO: THIS ONE SHOULD CHANGE PER LINE.
        }

        # TODO: WORK THIS OUT FROM THE ARGUMENTS PROVIDED
        channel_parameters = ["v_rad", "convolve", "c_0", "c_1", "c_2"]

        def minimize(theta, full_output=False):

            theta_dict = dict(zip(channel_parameters, theta))

            # The parameters in theta_dict (e.g., v_rad, convolve, etc) need to be
            # passed to the model so that they are applied to all synthesis regions
            model_kwargs.update(theta_dict)

            total_chi_sq = 0
            line_abundances = []
            for line_list_filename in line_list_filenames:

                line_abundance, chi_sq_i, warnflag = model.ApproximateModelSpectrum(
                    line_list_filename, **model_kwargs).optimise(spectrum)
                
                total_chi_sq += chi_sq_i
                line_abundances.append(line_abundance)
        
            if full_output:
                return (total_chi_sq, line_abundances)

            return total_chi_sq

        opt_channel_parameters = op.minimize(minimize, p0)
        chi_sq, line_abundances = minimize(opt_channel_parameters, full_output=True)

        return (opt_channel_parameters, line_abundances)



