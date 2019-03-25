import setuptools

setuptools.setup(
    name="hetdex-elixer",
    version="1.6.2",
    author="Dustin Davis",
    author_email="dustin.davis@utexas.edu",
    description="HETDEX Emission Line eXplorer tool",
    url="https://github.com/phantomamoeba/elixer.git",
    packages=setuptools.find_packages(),
	include_package_data=True,
    classifiers=["Development Status :: 4 - Beta",
                 "Environment :: Console",
                 "Intended Audience :: Developers",
                 "Intended Audience :: Science/Research",
                 "License :: OSI Approved :: GNU General Public License (GPL)",
                 "Operating System :: POSIX :: Linux",
                 "Programming Language :: Python :: 3.6",
                 "Topic :: Scientific/Engineering :: Astronomy",
                 "Topic :: Utilities",
                 ]

)
