import re

import setuptools
from setuptools import setup


with open('README.md', 'r') as f:
    long_description = f.read()
    descr_lines = long_description.split('\n')
    descr_no_gifs = []  # gifs are not supported on PyPI web page
    for l in descr_lines:
        if not ('<img src=' in l and 'gif' in l):
            descr_no_gifs.append(l)

    long_description = '\n'.join(descr_no_gifs)


setup(
    # Information
    name='sample-factory',
    description='High throughput asynchronous reinforcement learning framework',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    keywords='asynchronous reinforcement learning policy gradient ppo appo impala',

    # these requirements are untested and incomplete. Follow README.md to properly setup the environment.
    # Full set of tested requirements is in environment.yml
    install_requires=[
        'numpy>=1.18.1',
        'torch>=1.6',
        'gym>=0.17.1',
        'tensorboard>=1.15.0',
        'tensorboardx>=2.0',
        'psutil>=5.7.0',
        'threadpoolctl>=2.0.0',
        'colorlog',
        'faster-fifo>=1.0.9',
        'filelock',
        'opencv-python',
    ],

    package_dir={'': './'},
    packages=setuptools.find_packages(where='./', include='sample_factory*'),
    include_package_data=True,

    python_requires='>=3.6',
)
