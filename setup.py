#!/usr/bin/env python
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name="CATlas",
    version="0.0",
    description="Python Distribution Utilities",
    author="Brook Wander, Kirby Broderick, and others in the Ulissi research group",
    author_email="bwander@andrew.cmu.edu",
    url="https://github.com/ulissigroup/CATlas",
    packages=["catlas"],
)
