# coding: utf-8

""" Interface to MOOG(SILENT) """

__author__ = "Andy Casey <andy@astrowizici.st>"

__all__ = ["instance"]

# Standard library
import logging
import os
import re
import shutil
from operator import itemgetter
from random import choice
from signal import alarm, signal, SIGALRM, SIGKILL
from string import ascii_letters
from subprocess import PIPE, Popen
from textwrap import dedent

# Third party
import numpy as np

# Module specific
import utils

logger = logging.getLogger(__name__)

class instance(object):
    """ A context manager for dealing with MOOG """

    _executable = "MOOGSILENT"
    _acceptable_return_codes = (0, )

    def __init__(self, twd_base_dir="/tmp/", chars=10):
        """ Initialisation class allows the user to specify a base temporary
        working directory """

        self.chars = chars
        self.twd_base_dir = twd_base_dir
        if not os.path.exists(self.twd_base_dir):
            os.mkdir(self.twd_base_dir)

    def __enter__(self):
        # Create a temporary working directory
        self.twd = os.path.join(self.twd_base_dir, "".join([choice(ascii_letters) for _ in xrange(self.chars)]))
        while os.path.exists(self.twd):
            self.twd = os.path.join(self.twd_base_dir, "".join([choice(ascii_letters) for _ in xrange(self.chars)]))
        
        os.mkdir(self.twd)
        if len(self.twd) > 40:
            warnings.warn("MOOG has trouble dealing with absolute paths greater than 40 characters long. Consider"
                " a shorter absolute path for your temporary working directory.")
        return self


    def execute(self, filename=None, timeout=30, shell=False, env=None):
        """ Execute a MOOG input file with a timeout after which it will be forcibly killed. """

        if filename is None:
            filename = os.path.join(self.twd, "batch.par")

        logger.info("Executing MOOG input file: {0}".format(filename))

        class Alarm(Exception):
            pass

        def alarm_handler(signum, frame):
            raise Alarm

        if env is None and len(os.path.dirname(self._executable)) > 0:
            env = {"PATH": os.path.dirname(self._executable)}

        p = Popen([os.path.basename(self._executable)], shell=shell, bufsize=2056, cwd=self.twd, stdin=PIPE, stdout=PIPE, 
            stderr=PIPE, env=env, close_fds=True)

        if timeout != -1:
            signal(SIGALRM, alarm_handler)
            alarm(timeout)
        try:
            # Stromlo clusters may need a "\n" prefixed to the input for p.communicate
            pipe_input = "\n" if -6 in acceptable_moog_return_codes else ""
            pipe_input += os.path.basename(filename) + "\n"*100

            stdout, stderr = p.communicate(input=pipe_input)
            if timeout != -1:
                alarm(0)
        except Alarm:

            # process might have died before getting to this line
            # so wrap to avoid OSError: no such process
            try:
                os.kill(p.pid, SIGKILL)
            except OSError:
                pass
            return (-9, '', '')

        if p.returncode not in self._acceptable_return_codes:
            logger.warn("MOOG returned the following message:")
            logger.warn(stdout)
            logger.warn("MOOG returned the following errors (code: {0:d}):".format(p.returncode))
            logger.warn(stderr)

        return (p.returncode, stdout, stderr)


    def _format_ew_input(self, measurements, force_loggf=True, comment=None):
        """Writes equivalent widths to an output file which is compatible for
        MOOG.
        
        Parameters
        ----
        measurements : a record array
            A list where each row contains the following:
        
        filename : str
            Output filename for the measured equivalent widths.
        
        transitions : list of float-types, optional
            A list of transitions to write to file. Default is to write "all"
            transitions to file.

        force_loggf : bool
            Should MOOG treat all the oscillator strengths as log(gf) values?
            If all lines in the list have positive oscillator strengths, using
            this option will force a fake line in the list with a negative
            oscillator strength.
            
        comment : str, optional
            A comment to place at the top of the output file.
        
        clobber : bool, optional
            Whether to over-write the `filename` if it already exists.
        """
        
        output_string = comment.rstrip() if comment is not None else ""
        output_string += "\n"
        
        measurements_arr = []
        for i, measurement in enumerate(measurements):

            if transitions != "all" and measurement.transition not in transitions: continue
            if not measurement.is_acceptable or 0 >= measurement.measured_equivalent_width: continue
            
            measurements_arr.append([measurement.rest_wavelength,
                                     measurement.transition,
                                     measurement.excitation_potential,
                                     measurement.oscillator_strength,
                                     measurement.measured_equivalent_width])


        if force_loggf and np.all(measurements["loggf"] > 0):
            logger.warn("Adding fake line at 900 nm with a negative oscillator strength.")
            
            measurements = np.append(measurements, np.array([faux_line], dtype=measurements.dtype))
            measurements_arr = np.insert(measurements_arr, 0, [9000., 89., 0.0, -9.999, 0.0], axis=0)
        
        # Sort all the lines first transition, then by wavelength
        measurements = sorted(measurements, key=itemgetter("species", "wavelength"))

        for measurement in measurements:
            output_string += "{0:10.3f} {1:9.3f} {2:8.2f} {3:6.2f}                             {4:5.1f}\n".format(
                *[measurement[col] for col in ["wavelength", "species", "excitation_potential", "loggf", "equivalent_width"]])

        return output_string
        


    def _format_abfind_input(self, line_list_filename, model_atmosphere_filename, standard_out,
        summary_out, terminal="x11", atmosphere=1, molecules=0, truedamp=1, lines=1,
        freeform=0, flux_int=0, damping=0, units=0):

        output = """
        abfind
        terminal '{terminal}'
        standard_out '{standard_out}'
        summary_out '{summary_out}'
        model_in '{model_atmosphere_filename}'
        lines_in '{line_list_filename}'
        atmosphere {atmosphere}
        molecules {molecules}
        lines {lines}
        freeform {freeform}
        flux/int {flux_int}
        damping {damping}
        plot 0
        """.format(**locals())
        
        return dedent(output).lstrip()


    def _parse_abfind_summary_output(self, filename):
        """ Reads the summary output filename after MOOG's `abfind` has been
        called and returns a numpy record array """

        with open(filename, "r") as fp:
            output = fp.readlines()

        data = []
        columns = ("wavelength", "species", "excitation_potential", "loggf", "equivalent_width",
            "abundance")

        for i, line in enumerate(output):
            if line.startswith("Abundance Results for Species "):
                element, ionization = line.split()[4:6]
                current_species = utils.element_to_species("{0} {1}".format(element, ionization))
                
                # Check if we already had this species. If so then MOOG has run >1 iteration.
                if len(data) > 0:
                    exists = np.where(np.array(data)[:, 1] == current_species)

                    if len(exists[0]) > 0:
                        logger.debug("Detecting more than one iteration from MOOG")
                        data = list(np.delete(np.array(data), exists, axis=0))
                continue

            elif re.match("^   [0-9]", line):
                line_data = map(float, line.split())
                # Delete the logRW column
                del line_data[4]
                # Delete the del_avg column
                del line_data[-1] 

                # Insert a species column
                line_data.insert(1, current_species)
                data.append(line_data)
                continue

        return np.core.records.fromarrays(np.array(data).T,
            names=columns, formats=["f8"] * len(columns))


    def abfind(self, measurements, model_atmosphere, **kwargs):
        """ Call `abfind` in MOOG """

        if os.path.dirname(model_atmosphere) != self.twd:
            shutil.copy(model_atmosphere, self.twd)
            model_atmosphere = os.path.join(self.twd, os.path.basename(model_atmosphere))

        elif not os.path.exists(model_atmosphere):
            raise IOError("model atmosphere filename {0} does not exist".format(model_atmosphere))

        # Write the equivalent widths to file
        line_list_filename = os.path.join(self.twd, "ews")
        with open(line_list_filename, "w") as fp:
            fp.write(self._format_ew_input(measurements, **kwargs))

        # Prepare the input and output filenames
        input_filename, standard_out, summary_out = [os.path.join(self.twd, filename) \
            for filename in ("batch.par", "abfind.std", "abfind.sum")]
        
        # Write the abfind file
        with open(input_filename, "w") as fp:
            fp.write(self._format_abfind_input(line_list_filename, model_atmosphere, standard_out,
                summary_out, **kwargs))

        # Execute MOOG
        result, stdout, stderr = self.execute()

        # Parse the output
        return self._parse_abfind_summary_output(summary_out)


    def __exit__(self, type, value, traceback):
        # Remove the temporary working directory and any files in it
        shutil.rmtree(self.twd)
        return False



def abundance_differences(composition_a, composition_b, tolerance=1e-2):
    """Returns a key containing the abundance differences for elements that are
    common to `composition_a` and `composition_b`. This is particularly handy
    when scaling from one Solar composition to another.

    Inputs
    ----
    composition_a : `dict`
        The initial composition where elements are represented as keys and the
        abundances are inputted as values. The keys are agnostic (strings, floats,
        ints), as long as they have the same structure as composition_b.

    composition_b : `dict`
        The second composition to compare to. This should have the same format
        as composition_a

    Returns
    ----
    scaled_composition : `dict`
        A scaled composition dictionary for elements that are common to both
        input compositions."""

    tolerance = abs(tolerance)
    if not isinstance(composition_a, dict) or not isinstance(composition_b, dict):
        raise TypeError("Chemical compositions must be dictionary types")

    common_elements = set(composition_a.keys()).intersection(composition_b.keys())

    scaled_composition = {}
    for element in common_elements:
        if np.abs(composition_a[element] - composition_b[element]) >= tolerance:
            scaled_composition[element] = composition_a[element] - composition_b[element]

    return scaled_composition
   
