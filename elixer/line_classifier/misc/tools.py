"""

Handy functions that don't fit elsewhere

AUTHOR: Daniel Farrow (MPE)
"""

from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from scipy.interpolate import interp1d 

def generate_cosmology_from_config(config):
    """
    Generate an astropy.cosmology object
    from a config file
    """
    # Planck LCDM (from Ariel)
    hubble = config.getfloat("Cosmology", "H0")
    ombh2 = config.getfloat("Cosmology", "ombh2")
    omch2 = config.getfloat("Cosmology", "omch2")
    omnuh2 = config.getfloat("Cosmology", "omnuh2")
    Neff = config.getfloat("Cosmology", "Neff") # Standard model

    try:
        h = config.getfloat("Cosmology", "h_for_omegas")
    except:
        h = 0.67

    # Always divide by Planck cosmology as only want units in H=100
    _flc = FlatLambdaCDM(hubble, (omch2 + ombh2)/(h*h))  #ignore neutrinos

    return _flc


def read_flim_file(filename, random_flim_noise=None):
    """
    Read a file containing the flux limit, in the
    format of the Wiki flux limit predictions
    (from the Yorick code)

    Parameters
    ----------
    filename : str
        the filename of the flux limit file

    Returns
    -------
    flux_limits : callable
        a function that returns a flux limit
        when passed a redshift
    """

    # just apply the sensitivity estimates
    table_limit = Table.read(filename,
                       names=['redshift',  'vertex', 'centered1', 'centered7'],
                       format='ascii')

    flux_limits = interp1d(table_limit["redshift"], table_limit["centered1"], bounds_error=False, fill_value=0.0)

    return flux_limits
