# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages
import re

__version__ = ""
exec(open('transforms_templates/version.py').read())


# from fastai:
# helper functions to make it easier to list dependencies not as a python list,
# but vertically w/ optional built-in comments to why a certain version of the dependency is listed
def cleanup(x):
    return re.sub(r' *#.*', '', x.strip())  # comments


def to_list(buffer):
    return list(filter(None, map(cleanup, buffer.splitlines())))


# ## developer dependencies ###
#
# anything else that's not required by a user to run the library, but
# either is an enhancement or a developer-build requirement goes here.
#
# the [dev] feature is documented here:
# https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras-optional-features-with-their-own-dependencies
#
# these, including the normal dependencies, get installed with:
#
#   pip install "champs[dev]"
#
# or via an editable install:
#
#   pip install -e ".[dev]"
#
# some of the listed modules appear in test_requirements as well, as explained below.
#
dev_requirements = {'dev': to_list("""
    coverage                     # make coverage
    distro
    ipython
    jupyter
    jupyter_contrib_nbextensions
    nbconvert>=5.4
    nbdime                       # help with nb diff/merge
    nbformat
    notebook>=5.7.0
    pip>=9.0.1
    pipreqs>=0.4.9
    pytest
    pytest-xdist                 # make test-fast (faster parallel testing)
    responses                    # for requests testing
    traitlets
    wheel>=0.30.0
    graphviz
""")}

requirements = ['humanize', 'joblib', 'autopep8', 'matplotlib', 'pandas', 'tqdm', 'dataclasses','mmcv', 'addict', 'six']


# pyquaternion, tensorboardX scikit-image

setup(
    name='transforms_templates',
    version=__version__,
    description='Transforms templates (draft)',
    author='Alexey U. Gudchenko',
    author_email='proga@goodok.ru',
    url='https://github.com/goodok/transforms_templates',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=requirements,
    extras_require=dev_requirements,
)
