#!/usr/bin/env python
try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

setup(
    name="catlas",
    version="0.0",
    description="Python Distribution Utilities for enumerating adslabs and performing ML inference of energies",
    author="Brook Wander, Kirby Broderick, and others in the Ulissi research group",
    author_email="bwander@andrew.cmu.edu",
    url="https://github.com/ulissigroup/catlas",
    packages=find_packages(),
)
