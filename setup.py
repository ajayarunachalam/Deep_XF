#! /usr/bin/env python


# Copyright (C) 2021-2022 Ajay Arunachalam <ajay.arunachalam08@gmail.com>
# License: MIT, ajay.arunachalam08@gmail.com

import pip
import logging
import pkg_resources
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


def _parse_requirements(file_path):
    pip_ver = pkg_resources.get_distribution('pip').version
    pip_version = list(map(int, pip_ver.split('.')[:2]))
    if pip_version >= [6, 0]:
        raw = pip.req.parse_requirements(file_path,
                                         session=pip.download.PipSession())
    else:
        raw = pip.req.parse_requirements(file_path)
    return [str(i.req) for i in raw]


# parse_requirements() returns generator of pip.req.InstallRequirement objects
try:
    install_reqs = _parse_requirements("requirements.txt")
except Exception:
    logging.warning('Fail load requirements file, so using default ones.')
    install_reqs = []

__version__ = '0.0.1'

def readme():
    with open('README.rst', 'r', encoding='utf-8') as f:
        return f.read()


setup(
    name='deep_xf',
    version=__version__,
    packages=["deep_xf"],
    description='DEEPXF - An open-source, low-code explainable forecasting and nowcasting library with state-of-the-art deep neural networks and Dynamic Factor Model.',
    long_description = readme(),
    long_description_content_type="text/markdown",
    url='https://github.com/ajayarunachalam/Deep_XF',
    install_requires=install_reqs,
    license='MIT',
    include_package_data=True,
    author='Ajay Arunachalam',
    author_email='ajay.arunachalam08@gmail.com')
