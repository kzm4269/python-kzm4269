#!/usr/bin/env python
import sys

from setuptools import setup, find_packages

if sys.version_info < (3, 6):
    sys.exit('Sorry, Python < 3.6 is not supported')

with open('README.md') as f:
    README = f.read()

with open('LICENSE') as f:
    LICENSE = f.read()

setup(
    name='kzm4269',
    version='0.0',
    description="kzm4269's Python utilities",
    long_description=README,
    author='kzm4269',
    author_email='4269kzm@gmail.com',
    url='https://github.com/kzm4269/python-kzm4269',
    license=LICENSE,
    packages=find_packages(exclude=['tests']),
    test_suite='tests',
)
