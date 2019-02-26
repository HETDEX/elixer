HETDEX ELiXer (Emission Line eXplorer) Data Release 1 Read Me


##############################
# Installation
##############################

ELiXer requires no installation itself, but does have a number of dependencies. The recommended setup follows
immediately with alternatives after.

* RECOMMENDED SETUP

It is recommended that you add the following lines to your .bashrc file





* ALTERNATE SETUP

Alternatively, you may use your own python installation and either run ELiXer from the HDR1 directory as above or copy
the ELiXer folder to your own location and run from there.

ELiXer requires python 2.7.15 (due to certain package dependencies, python 3+ is not supported at this time)
Additional (common) python packages (should be included in python 2.7.15 w/o additional installation)
    numpy, matplotlib, pylab, argparse, sys, os, distutils, glob, shutils, socket

Additional (less common) packages: (install with "pip install --user xxx"  where xxx is the package name)
 astropy, pandas, photutils, scipy, tables (aka pytables)


One additional HETDEX specific package is also required: pyhetdex
    ("pip install --extra-index-url https://gate.mpe.mpg.de/pypi/simple/ pyhetdex")



##############################
# Usage
##############################

SLURM vs Single Instance

Common commandlines




##############################
# Output
##############################

What to expect