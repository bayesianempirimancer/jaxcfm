#!/usr/bin/env python

import os

from setuptools import find_packages, setup

install_requires = [
    "jax>=0.4.0",
    "jaxlib>=0.4.0",
    "flax>=0.7.0",
    "optax>=0.1.0",
    "diffrax>=0.4.0",
    "matplotlib",
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "scikit-learn>=1.0.0",
    "pot>=0.9.0",  # For testing/comparison purposes
    "absl-py>=0.15.0",
    "tqdm>=4.62.0",
]

version_py = os.path.join(os.path.dirname(__file__), "jaxcfm", "version.py")
version = open(version_py).read().strip().split("=")[-1].replace('"', "").strip()
readme = open("README.md", encoding="utf8").read()
setup(
    name="jaxcfm",
    version=version,
    description="JAX/Flax Implementation of Conditional Flow Matching for Fast Continuous Normalizing Flow Training.",
    author="JAXCFM Contributors",
    author_email="",  # Update with appropriate contact email
    url="https://github.com/bayesianempirimancer/jaxcfm",  # Update with actual repository URL
    install_requires=install_requires,
    license="MIT",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.8",
)
