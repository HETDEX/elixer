from setuptools import setup, find_packages

install_requires = ['numpy>=1.18.2', 'astropy>=5.0', 'scipy>=1.6',
                    'tables>=3.8', 'speclite>=0.16', 'emcee>=3.1',
                    'photutils>=1.8','astroquery>=0.3.10','pandas>=1.3',
                    'pdf2image>=1.16.0', 'sep>=1.2',
                    'specutils>=1.11']

###########################################################################################################
#note: also needs pyhetdex >= 0.14, but it has an unusual install and causes
#problems here
#to install manually: pip install --extra-index-url https://gate.mpe.mpg.de/pypi/simple/ pyhetdex
###########################################################################################################

extras = {}

setup(
    name="elixer",
    version="1.22.0",
    author="Dustin Davis",
    author_email="dustin.davis@utexas.edu",
    description="HETDEX Emission Line eXplorer tool",
    url="https://github.com/HETDEX/elixer.git",
    packages=find_packages(),
    include_package_data=True,
    data_files=[('.',['elixer/distance_list_bool.txt',]),],
    zip_safe=False,
    install_requires=install_requires,
    extras_require=extras,
    entry_points={},
    classifiers=["Development Status :: 4 - Beta",
                 "Environment :: Console",
                 "Intended Audience :: Developers",
                 "Intended Audience :: Science/Research",
                 "License :: OSI Approved :: GNU General Public License (GPL)",
                 "Operating System :: POSIX :: Linux",
                 "Programming Language :: Python :: 3.7",
                 "Topic :: Scientific/Engineering :: Astronomy",
                 "Topic :: Utilities",
                 ]
)

print("You may need to manually install pyhetdex")
print("run: pip install --extra-index-url https://gate.mpe.mpg.de/pypi/simple/ pyhetdex ")

