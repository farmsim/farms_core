#!/usr/bin/env python
""" Setup script """

from setuptools import setup, find_packages

setup(
    name='farms_data',
    version='0.1',
    author='farmsdev',
    author_email='biorob-farms@groupes.epfl.ch',
    description='FARMS package for data handling',
    # license='BSD-3',
    keywords='farms data simulation',
    # url='',
    # packages=['farms_data'],
    packages=find_packages(),
    # long_description=read('README'),
    # classifiers=[
    #     'Development Status :: 3 - Alpha',
    #     'Topic :: Utilities',
    #     'License :: OSI Approved :: BSD License',
    # ],
    scripts=[],
    # package_data={'farms_data': [
    #     'farms_data/templates/*',
    #     'farms_data/config/*'
    # ]},
    include_package_data=True,
    zip_safe=False,
    # install_requires=[
    #     'cython',
    #     'numpy',
    #     'trimesh',
    #     'pydata'
    # ],
)
