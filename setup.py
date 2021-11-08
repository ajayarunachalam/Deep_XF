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


__version__ = '0.0.5'

def readme():
    with open('README.rst', 'r', encoding='utf-8') as f:
        return f.read()


setup(
    name='deep_xf',
    version=__version__,
    packages=["deep_xf"],
    description='DEEPXF - An open-source, low-code explainable forecasting and nowcasting library with state-of-the-art deep neural networks and Dynamic Factor Model. Now available with additional addons like Denoising TS signals with ensembling of filters, TS signal similarity test with Siamese Neural Networks',
    long_description = readme(),
    long_description_content_type="text/markdown",
    url='https://github.com/ajayarunachalam/Deep_XF',
    install_requires=[
        "ipython",
        "jupyter",
        "tqdm",
        "pandas",
        "matplotlib",
        "seaborn==0.9.0",
        "scikit-learn>=0.24.0",
        "pandas_profiling",
        "statsmodels==0.12.2",
        "keras",
        "torch",
        "shap==0.39.0",
        "py-ecg-detectors",
    ],
    license='MIT',
    include_package_data=True,
    author='Ajay Arunachalam',
    author_email='ajay.arunachalam08@gmail.com')
