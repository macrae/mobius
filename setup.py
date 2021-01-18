#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

def parse_requirements(file):
    with open(file, "r") as fs:
        return [r for r in fs.read().splitlines() if
                (len(r.strip()) > 0 and not r.strip().startswith("#") and not r.strip().startswith("--"))]

requirements = parse_requirements('requirements.txt')

test_requirements = parse_requirements('requirements-dev.txt')

setup_requirements = ['pytest-runner',]

setup(
    author="Sean MacRae",
    author_email='s.mac925@gmail.com',
    python_requires='>=3.6',
    description="Methods for metric-learning tabular data",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='mobius',
    name='mobius',
    packages=find_packages(include=['mobius', 'mobius.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/macrae/mobius',
    version='0.1.0',
    zip_safe=False,
)
