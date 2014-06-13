# coding: utf-8

""" Utility functions """

from __future__ import division, print_function

__author__ = "Andy Casey <andy@astrowizici.st>"
__all__ = ["element_to_species", "species_to_element", "get_common_letters", \
    "find_common_start", "extend_limits", ]

# Third party imports
import numpy as np

def element_to_species(element_repr):
    """
    Return the floating point representation for an element (atomic number +
    ionization state/10)
    """
    
    periodic_table = """H                                                  He
                        Li Be                               B  C  N  O  F  Ne
                        Na Mg                               Al Si P  S  Cl Ar
                        K  Ca Sc Ti V  Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
                        Rb Sr Y  Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I  Xe
                        Cs Ba Lu Hf Ta W  Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn
                        Fr Ra Lr Rf Db Sg Bh Hs Mt Ds Rg Cn UUt"""
    
    lanthanoids    =   "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb"
    actinoids      =   "Ac Th Pa U  Np Pu Am Cm Bk Cf Es Fm Md No"
    
    periodic_table = periodic_table.replace(" Ba ", " Ba " + lanthanoids + " ") \
        .replace(" Ra ", " Ra " + actinoids + " ").split()
    del actinoids, lanthanoids
    
    if not isinstance(element_repr, (unicode, str)):
        raise TypeError("element must be represented by a string-type")
        
    if element_repr.count(" ") > 0:
        element, ionization = element_repr.split()[:2]
    else:
        element, ionization = element_repr, "I"
    
    if element not in periodic_table:
        # Don"t know what this element is
        return float(element_repr)
    
    ionization = max([0, ionization.upper().count("I") - 1]) /10.
    transition = periodic_table.index(element) + 1 + ionization
    return transition


def species_to_element(species):
    """
    Return the string representation for a species.
    """
    
    periodic_table = """H                                                  He
                        Li Be                               B  C  N  O  F  Ne
                        Na Mg                               Al Si P  S  Cl Ar
                        K  Ca Sc Ti V  Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
                        Rb Sr Y  Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I  Xe
                        Cs Ba Lu Hf Ta W  Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn
                        Fr Ra Lr Rf Db Sg Bh Hs Mt Ds Rg Cn UUt"""
    
    lanthanoids    =   "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb"
    actinoids      =   "Ac Th Pa U  Np Pu Am Cm Bk Cf Es Fm Md No"
    
    periodic_table = periodic_table.replace(" Ba ", " Ba " + lanthanoids + " ") \
        .replace(" Ra ", " Ra " + actinoids + " ").split()
    del actinoids, lanthanoids
    
    if not isinstance(species, (float, int)):
        raise TypeError("species must be represented by a floating point-type")
    
    if species + 1 >= len(periodic_table) or 1 > species:
        # Don't know what this element is. It"s probably a molecule.
        common_molecules = {
            112: ["Mg", "H"],
            606: ["C", "C"],
            607: ["C", "N"],
            106: ["C", "H"],
            108: ["O", "H"]
        }
        if species in common_molecules.keys():
            elements_in_molecule = common_molecules[species]
            if len(list(set(elements_in_molecule))): return "{0}_{1}".format(
                elements_in_molecule[0], len(elements_in_molecule))

            return "-".join(elements_in_molecule)

        else:
            # No idea
            return str(species)
        
    atomic_number = int(species)
    element = periodic_table[int(species) - 1]
    ionization = int(round(10 * (species - int(species)) + 1))

    return "{0} {1}".format(element, "I" * ionization)


def get_common_letters(strlist):
    """
    Find the common letters in a list of strings.
    """
    return "".join([x[0] for x in zip(*strlist) \
        if reduce(lambda a,b:(a == b) and a or None,x)])


def find_common_start(strlist):
    """
    Find the common starting characters in a list of strings.
    """

    strlist = strlist[:]
    prev = None
    while True:
        common = get_common_letters(strlist)
        if common == prev:
            break
        strlist.append(common)
        prev = common

    return get_common_letters(strlist)


def extend_limits(values, fraction=0.10, tolerance=1e-2):
    """
    Extend the values of a list by a fractional amount.
    """

    values = np.array(values)
    finite = np.isfinite(values)

    if np.sum(finite) == 0:
        raise ValueError("no finite values provided")

    lower_limit, upper_limit = np.min(values[finite]), np.max(values[finite])
    ptp_value = np.ptp([lower_limit, upper_limit])

    new_limits = (lower_limit - fraction * ptp_value,
        ptp_value * fraction + upper_limit)

    if np.abs(new_limits[0] - new_limits[1]) < tolerance:
        if np.abs(new_limits[0]) < tolerance:
            # Arbitrary limits, since we"ve just been passed zeros
            offset = 1

        else:
            offset = np.abs(new_limits[0]) * fraction
            
        new_limits = new_limits[0] - offset, offset + new_limits[0]

    return np.array(new_limits)

