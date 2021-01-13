# -*- coding: utf-8 -*-
"""
Setup for repo template.

The project contains a **single top-level package**.
The top-level package package.py stores all the project info.
"""

import codecs
from os import path

from setuptools import setup

kwargs = dict(
    name='mmdemux',
    version=0.1,
    description='',
    author='Simone Marsili',
    author_email='simo.marsili@gmail.com',
    license='MIT',
    url='https://github.com/simomarsili/mmdemux',
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 1 - Planning',
        'License :: OSI Approved :: MIT',
        'Programming Language :: Python :: 3',
    ])


def get_long_description(readme):
    """Get the long description from the README file."""
    with codecs.open(readme, encoding='utf-8') as _rf:
        return _rf.read()


base_dir = path.abspath(path.dirname(__file__))
readme_file = path.join(base_dir, 'README.md')
long_description = get_long_description(readme_file)

modules = ['mmdemux']
packages = []

SETUP_REQUIRES = []
INSTALL_REQUIRES = [
    # # this is an example of URL based requirement (see PEP508):
    # 'repo @ http://github.com/user/repo/archive/master.tar.gz',
]
EXTRAS_REQUIRES = {'test': ['pytest']}

setup(long_description=long_description,
      py_modules=modules,
      packages=packages,
      entry_points={'console_scripts': ['mmdemux=mmdemux:main']},
      setup_requires=SETUP_REQUIRES,
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRES,
      **kwargs)
