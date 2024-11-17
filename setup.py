# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages
import sys
import os
import pip

print("install package ...\n", find_packages())

with open('requirements.txt') as f:
    requirements = f.readlines()

pip.main(["install", "gradio==4.44.0"])

setup(
    name="auto_openai",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
)
