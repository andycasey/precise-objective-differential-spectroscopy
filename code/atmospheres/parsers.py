# coding: utf-8

""" Parsers for stellar atmosphere models. """

__author__ = "Andy Casey <andy@astrowizici.st>"

# Standard libraries
import os
from textwrap import dedent

# Third party libraries
import numpy as np

__all__ = ["CastelliKuruczAlphaParser"]

class CastelliKuruczAlphaParser(object):
    """
    A class to parse Castelli & Kurucz alpha-enhanced (ODFNEW) model atmospheres.
    """

    ntau = 72
    parameters = ["Teff", "logg", "[M/H]"]
    atmosphere_parameters = ["RHOX", "T", "P", "XNE", "ABROSS", "ACCRAD", "VTURB",
        "FLXCNV", "VCONV", "VELSND"]

    @staticmethod
    def filename(filename):

        # This should really be a regular expressions match.
        filename = os.path.basename(filename)
        
        feh = float(filename.split('t')[0][1::].replace('m', '-').replace('p', '+').rstrip('a')) /10.
        teff = float(filename.split('t')[1].split('g')[0])
        logg = float(filename.split('g')[1].split('k')[0]) /10.
        
        return [teff, logg, feh]

    @staticmethod
    def contents(filename):
        
        with open(filename, 'r') as fp:
            contents = fp.readlines()
        
        in_deck, deck = False, []
        
        for line in contents:
            if line.startswith('READ DECK'):
                in_deck = True
                continue
            elif line.startswith('PRADK'):
                break
            
            if in_deck: deck.append(map(float, line.split()))
            
        return np.array(deck)

    @staticmethod
    def write(point, xi, photosphere, output_filename, **kwargs):
        
        teff, logg, feh = point
        output_string = """
        KURUCZ
                  TEFF   %i.  GRAVITY %2.5f LTE
        NTAU        %i
        """ % (teff, logg, len(photosphere), )
        
        output_string = dedent(output_string).lstrip()
        
        for line in photosphere:
            # Need to accompany thermal structures of different array sizes
            output_string += " %1.8e " % (line[0], )
            output_string += ("%10.3e" * (len(line[1:])) % tuple(line[1:]) + "\n")

        output_string += "         %1.2f\n" % (xi, )
        output_string += "NATOMS        0     %2.1f\n" % (feh,)
        output_string += "NMOL          0\n"

        # Add the trailing newline character
        output_string += "\n"

        # Write it out
        with open(output_filename, 'w') as fp:
            fp.write(output_string)

        return True
