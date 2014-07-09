# coding: utf-8

""" Stellar atmosphere model interpolator. """

__author__ = "Andy Casey <andy@astrowizici.st>"

# Standard libraries
import os
import logging
from glob import glob

# Third party libraries
import numpy as np
import scipy.interpolate

# Module-specific
import parsers

__all__ = ["Interpolator"]

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Interpolator(object):
    """
    An object to represent a regular, or irregular grid of model atmospheres.
    """

    _parsers_ = {
        "Castelli & Kurucz (2003, NEWODF, [alpha/Fe] = +0.4)": parsers.CastelliKuruczAlphaParser
    }

    def __init__(self, filename_wildmask):
        """
        Initiate the class.
        """

        self.filenames = glob(filename_wildmask)
        if len(self.filenames) == 0:
            raise IOError("no model files found")
        self.parse = self._parsers_[self.description]()
        self.parameters = self.parse.parameters
        self.points = np.array(map(self.parse.filename, self.filenames))
        self.boundaries = [(min(self.points[:, i]), max(self.points[:, i])) \
            for i in range(len(self.parameters))]

        assert len(self.parameters) == self.points.shape[1]
        # [TODO] Check for double-rows.


    @property
    def description(self):
        """
        Identify the type of model atmospheres from the contents of the first model.
        """

        # [TODO] Add more parsers.
        return "Castelli & Kurucz (2003, NEWODF, [alpha/Fe] = +0.4)"


    def _interpolate_photosphere(self, point, neighbours=1):
        """
        Interpolate the photospheric properties of a model atmosphere.
        """

        point = np.array(point)
        for value, boundary in zip(point, self.boundaries):
            if not (boundary[1] >= value >= boundary[0]):
                raise ValueError("Point requested is outside the boundaries.")
        
        # Check for an exact match.
        exact_match = np.where((self.points == point).all(axis=1))[0]
        if len(exact_match) == 1:
            return self.parse.contents(self.filenames[exact_match])
        elif len(exact_match) > 1:
            message = "Found multiple exact matches in atmospheric gridpoints:\n"
            message += "\n".join(["{0} ({1}) has values {2}".format(i, self.filenames[i], self.points[i]) \
                for i in exact_match])
            raise ValueError(message)
            
        # No exact match found; interpolation is required. Find the nearest points.
        indices = set(range(len(self.filenames)))
        for i, interp_value in enumerate(point):
            differences = self.points[:, i] - interp_value
            if (differences == 0).all(): continue

            idx = list(np.where(self.points[:, i] == interp_value)[0])
            if len(idx) == 0:
                pos_differences = np.sort(differences[np.where(differences>0)])[:neighbours]
                neg_differences = np.sort(differences[np.where(differences<0)])[-neighbours:]

                for each in pos_differences:
                    idx.extend(list(np.where(differences == each)[0]))
                for each in neg_differences:
                    idx.extend(list(np.where(differences == each)[0]))
            indices = indices.intersection(idx)

        indices = list(indices)
        
        # Generate the subgrid of points and values.
        subset_points = self.points[indices]
        subset_values = np.array([self.parse.contents(self.filenames[index]) for index in indices])

        # Protect from QHull errors.
        superfluous_columns = []
        for column in xrange(subset_points.shape[1]):
            if len(np.unique(subset_points[:, column])) == 1:
                superfluous_columns.append(column)
        interpolated_point = scipy.delete(point, superfluous_columns, axis=0)        
        subset_points = scipy.delete(subset_points, superfluous_columns, axis=1)

        # Re-scale subset_values onto a common optical depth scale.
        # [TODO]
        #common_tau_ross = subset_values[0, :, 0]
        #min_tau_ross, max_tau_ross = np.min(subset_points[:, :, 0]), np.max(subset_points[:, :, 0])
        #common_tau_ross = 

        return scipy.interpolate.griddata(subset_points, subset_values, interpolated_point.reshape(1, -1))[0]


    def __call__(self, point, output_filename, xi=0.0, clobber=False, **kwargs):
        """
        Interpolate a model atmosphere to the requested point, and write it to disk.
        """
        
        if os.path.exists(output_filename) and not clobber:
            raise OSError("output filename exists and we won't clobber it.")

        photosphere = self._interpolate_photosphere(point)
        return self.parse.write(point, xi, photosphere, output_filename,
            **kwargs)
        

