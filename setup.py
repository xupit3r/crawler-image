# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='crawler-image',
    version='1.0.0',
    description='Some image classification stuff for my crawler',
    long_description=readme,
    author='Joe D\'Alessandro',
    author_email='joe@thejoeshow.net',
    url='https://github.com/xupit3r/crawler-image',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

