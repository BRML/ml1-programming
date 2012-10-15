#! /usr/bin/env python
# -*- coding: utf-8 -*-


__author__ = 'Justin Bayer, bayer.justin@googlemail.com'


from setuptools import setup, find_packages


setup(
    name="ml-programming",
    version="pre-0.1",
    description=("programming assignments for machine learning course at"
                 " TUM informatics"),
    packages=find_packages(exclude=['examples', 'docs']),
    include_package_data=True,
)
