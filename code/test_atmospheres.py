# coding: utf-8

""" Test the model atmosphere interpolator """

__author__ = "Andy Casey <andy@astrowizici.st>"

import os
import logging

import numpy as np
import matplotlib.pyplot as plt

from atmosphere import  AtmosphereInterpolator as atmospheres, \
                        parsers 

def test_MARCS_atmospheres():

    atmosphere_type = "MARCS (2011)"
    solar_model_filename = "sun.mod.gz"

    if atmosphere_type not in parsers \
    or not os.path.exists(solar_model_filename):
        # Skip
        return None

    folder, parser = parsers[atmosphere_type]
    interpolator = atmospheres(folder, parser())

    # Parse the solar atmosphere
    solar_thermal_structure = interpolator.parser.parse_thermal_structure(
        os.path.join(folder, solar_model_filename))

    # These were taken directly from sun.mod.gz
    truths = [5777, np.log10(2.7542e+04), 0., 0.]

    # Interpolate the thermal structure for the truth values
    interpolated_thermal_structure = interpolator.interpolate_thermal_structure(*truths)

    # Generate comparisons
    x = "k"
    interpolated_properties = set(interpolated_thermal_structure.dtype.names).difference(x)

    K = int(np.ceil(len(interpolated_properties)**0.5))
    fig, axes = plt.subplots(K, K)

    for i, (ax, y) in enumerate(zip(axes.flatten(), interpolated_properties)):

        #ax.plot(solar_thermal_structure[x], solar_thermal_structure[y], 'k')
        #ax.plot(interpolated_thermal_structure[x], interpolated_thermal_structure[y], 'b')

        # Ensure the relative difference is less than 5%
        relative_difference = 100 * (solar_thermal_structure[y] - interpolated_thermal_structure[y])/solar_thermal_structure[y]
        finite = np.isfinite(relative_difference)
        if not np.all(relative_difference[finite] < 5.):
            logging.warn("Relative difference in {0} exceeds 5% ({1} > 5)!".format(y, int(np.max(relative_difference[finite]))))

        ax.plot(solar_thermal_structure[x], relative_difference, 'k')

        ax.set_xlabel(x)
        ax.set_ylabel(y)

    [each.set_visible(False) for each in axes.flatten()[len(interpolated_properties):]]

