from setuptools import setup, find_packages

install_requires = ['numpy>=1.18.2', 'astropy>=4.0', 'scipy>=1.4',
                    'tables>=3.6.1', 'speclite>=0.8', 'emcee>=3.0.2',
                    'photutils>=0.7.1','astroquery>=0.3.10','pandas>=1.1.0',
                    'pdf2image>=1.16.0', 'pyhetdex>=0.10', 'sep>=1.0.3',
                    'specutils>=1.0']

extras = {}

setup(
    name="elixer",
    version="1.13.0",
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
