"""Setup script for the pnslib package.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

from setuptools import setup
from setuptools import find_packages

classifiers = """
Development Status :: 2 - Pre-Alpha
Intended Audience :: Science/Research
Natural Language :: English
Operating System :: OS Independent
Programming Language :: Python :: 2.7
Programming Language :: Python :: 3.4
Programming Language :: Python :: 3.5
Topic :: Utilities
Topic :: Scientific/Engineering
Topic :: Scientific/Engineering :: Image Recognition
Topic :: Scientific/Engineering :: Computer Vision
Topic :: Scientific/Engineering :: Machine Learning
Topic :: Software Development :: Libraries :: Python Modules
License :: OSI Approved :: MIT License
"""

__version__ = "0.1.0-alpha.1"
__author__ = "P&S Team"
__url__ = "https://github.com/PnS2018/pnslib"

setup(
    name='pnslib',
    version=__version__,

    author=__author__,

    url=__url__,

    install_requires=["numpy",
                      "scipy",
                      "cv2",
                      "future"],
    extras_require={
          "h5py": ["h5py"],
      },

    packages=find_packages(),

    classifiers=list(filter(None, classifiers.split('\n'))),
    description="P&SLib - Utilities for P&S module projects and demos."
)
