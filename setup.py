from setuptools import setup, find_packages

install_requires = ['numpy', 'astropy', 'scipy',
                    'tables', 'speclite', 'emcee', 'photutils>=0.7.1',
                    'pdf2image>=1.9.0', 'pyhetdex', 'sep>=1.0.3',
                    'specutils>=0.7']

extras = {}

setuptools.setup(
    name="elixer",
    version="1.8.2",
    author="Dustin Davis",
    author_email="dustin.davis@utexas.edu",
    description="HETDEX Emission Line eXplorer tool",
    url="https://github.com/HETDEX/elixer.git",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    extras_require=extras,
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
