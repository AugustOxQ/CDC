#!/usr/bin/env python

import platform

from setuptools import find_namespace_packages, find_packages, setup

DEPENDENCY_LINKS = []
if platform.system() == "Windows":
    DEPENDENCY_LINKS.append("https://download.pytorch.org/whl/torch_stable.html")


def fetch_requirements(filename):
    with open(filename) as f:
        return [ln.strip() for ln in f.read().split("\n")]


setup(
    name="DeepClustering",
    version="0.0.1",
    description="Deep Clustering with Web-Scale Datasets",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="A",
    author_email="A",
    url="A",  # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    install_requires=fetch_requirements("requirements.txt"),
    # packages=find_packages(where="src"), # After publishing to PyPI, this should be added
    # package_dir={"src": "src"}, # After publishing to PyPI, this should be added
    python_requires=">=3.7.0",
    include_package_data=True,
    dependency_links=DEPENDENCY_LINKS,
    zip_safe=False,
)
