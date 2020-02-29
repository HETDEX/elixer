HETDEX ELiXer (Emission Line eXplorer) Data Release 1 Readme
------------------------------------------------------------

The following readme covers the installation/setup and basic usage of ELiXer as incorporated into the HETDEX
Data Release 1. This version of ELiXer was written for and tested with Python 2.7.15. Specifc APIs are compatible with
Python 3x (limited testing performed with Python 3.6.8) but there are deprication warnings for several packages
(notably astropy and photutils), however the PDF report generation (calls to elixer.py and selixer.py) MUST use
Python 2.7 due to several compatibility issues that will be addressed in a later release.


##############################
# Installation
##############################

ELiXer requires no installation itself, but does have a number of dependencies. The recommended setup follows
immediately with alternatives after.

* RECOMMENDED SETUP

It is recommended that you add the following line to your .bashrc file:

export PATH="/home/00115/gebhardt/anaconda2/bin:/home/00115/gebhardt/bin:/work/03946/hetdex/hdr1/software/elixer/bin:$PATH"

This will provide you with a common python and easy access to ELiXer bash wrappers.


* ALTERNATE SETUP

Alternatively, you may use your own python installation and either run ELiXer from the HDR1 directory as above or copy
the ELiXer folder to your own location and run from there.

ELiXer requires python 2.7.15 (due to certain package dependencies, python 3+ is not supported at this time)
Additional (common) python packages (should be included in python 2.7.15 w/o additional installation)
    numpy, matplotlib, pylab, argparse, sys, os, distutils, glob, shutils, socket

Additional (less common) packages: (install with "pip install --user xxx"  where xxx is the package name)
 astropy, ConfigParser, pandas, photutils, scipy, tables (aka pytables)


One additional HETDEX specific package is also required: pyhetdex
    ("pip install --user --extra-index-url https://gate.mpe.mpg.de/pypi/simple/ pyhetdex")



##############################
# Usage
##############################

ELiXer is intended to be run in one of two modes ... single instance and batch (via SLURM). If you are running on
your own desktop, you must use the single instance mode, but if you are running on a TACC cluster (maverick, wrangler,
or stampede2) you should normally use the batch or SLURM mode (TACC discourages the execution of high cost tasks from
the login nodes).

The basic command line for the two modes are almost identical, with the SLURM mode supporting a few additional
parameters related to the SLURM mechanism.

In the single instance mode, ELiXer will serially process each provided detection. In the SLURM mode, ELiXer will
spawn multiple instances of itself, dividing the detections across the instances, but still processing serially within
any given instance.

Run time varies with server, load, and proximity to the data, but you can generally assume approximately one minute
per detection report.

You may launch the ELiXer process with a python call: e.g.
(single instance version)
> python /work/03946/hetdex/hdr1/software/elixer/elixer.py --help
-or-
(SLURM version)
> python /work/03946/hetdex/hdr1/software/elixer/selixer.py --help

-OR- via the bash wrappers (the path to which were included in the line added to the .bashrc file)
> elixer --help
 -OR-
> selixer --help

The "--help" switch will print to screen a simple break out of the command line options. Also NOTE that the SLURM
version will NOT spawn multiple instances if the "--help" switch is on the command line (in this case, elixer and
selixer are equivalent (the other case is with the --merge switch described at the end of this readme).

* COMMON USAGE

Although there are many options, ELiXer is anticipated to be used in only a few ways with HETDEX Data Release 1.

Essentially, you will either provide an RA, Dec and search radius or a list of detection IDs and ELiXer will produce
a report for each.

The following examples will assume the SLURM version using the bash wrapper.

* EXAMPLE 1
> selixer --recover --ra 150.025406 --dec 2.087600 --error 2.0 --name example1 --tasks 0 --email yourname@utexas.edu

Here an --ra and --dec are provide (in decimal degress ... however, hms and dms notations will also work, e.g.:
 --ra 10h00m6.10s --dec 2d05m15.36s are equivalent).

  --recover is a switch that instructs ELiXer to run each detection to completion before starting the next one. This
            allows ELiXer to be run a second time, using the exact same commmand, if the first command timed out in
            the queue and it will resume where it left off and only process detections for which a report does not yet
            exist.
  --error is the radius in which to search from the given RA and Dec and is ALWAYS in arcsecs.
  --name is the output directory name under which the results will be written.
  --tasks 0 specifies that ELiXer should set the number of instances to spawn based on the cluster on which it
            is running. You may override this and supply a non-zero value to force ELiXer to use no more than that
            number of tasks.
  --email is entirely optional, but will generate emails to the supplied address when the SLURM job actually
          begins (when it exits the wait queue) and when it completes or ends via error.

Additional, optional commonly used command line switches:
    --time : supply a maximum hh:mm:ss runtime for the SLURM job (if not supplied, ELiXer will calculate a value)
    --queue : specify which queue to use to execute the SLRUM job on the cluster (if not supplied, ELiXer will
              choose a queue)


* Example 2
> selixer --recover --tasks 0 --email yourname@utexas.edu --dets detlist --error 2.5 --name example2

Here, --dets detlist refers to a file named detlist that contains a list of detectionIDs, one per line.

Obviously, you need to know the detectionIDs in advance (which may well be the case if you are already working with
them). This will actually allow detectids (like: 1000000318) OR inputids (in Datevshot_# format, like: 20180123v009_4).
OR all shots for a particular observation (like: 20180123v009)


This may also be a comma separated list like:
  --dets 1000000318,1000000330,1000000414
  --dets 20180123v009_4,1000000371
  (mixed types are allowed)




##############################
# Output
##############################

If you run the single instance of ELiXer, all the output will be immediately under a directory named for the
--name switch.

If you run the SLURM version of ELiXer, there will be a series of dispatch_xxx directories (where xxx is a number)
under the directory named for the --name switch and under each of those will be a log file and a directory named for
the datevshot of the detection. Under this second nested directory will be the output files for the detections.

The output files (excluding files created due to other options that may be supplied on the command line) at a
minimum, will consist of a report PDF (named after the detection) and two catalog files, also named after the
detection and terminated with *_cat.txt and *_fib.txt

The PDF reports graphically represent the basic information about the detection including information on the fiber
2D spectra, the fit of the emission line, the full summed/weighted spectra, photometric imaging (if available) and
potential catalog matches (if available).

The two catalogs summarize this data in text catalog format with the *_fib.txt file covering each fiber in the
HETDEX data and the *_cat.txt catalog focusing on the photometric catalog matches. Each begins with a header than
describe the columns.

The catalogs are fragmented under each of the dispatch_xxx directories, but you can combine them all into one
catalog (each) by running selixer --merge (or elixer --merge) at the parent directory of the dispatch_xxx folders.

