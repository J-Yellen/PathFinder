"""Setup file for pathfinder."""
from pathlib import Path
from setuptools import find_packages
from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

if Path("full_version.txt").is_file():
    with open("full_version.txt", "r", encoding="utf-8") as fh:
        """
        Note that this file is generated by the CI chain based on the git tag.
        It should not be present in the repository by default.
        """
        version_number = fh.read()
else:
    version_number = 'v0.0.0'  # default value when under development

setup(
    name='pathfinder',
    version=version_number,
    description='Tools for finding the optimum combination of features given pairwise relations and weights',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Jamie Yellen',
    url='https://github.com/J-Yellen/PathFinder',
    python_requires=">=3.9",
    packages=find_packages(),
    # note requirements listed ininstall_requires should be the *minimum required*
    # in order to allow pip to resolve multiple installed packages properly.
    # requirements.txt should contain a specific known working version instead.
    install_requires=[
        'ipykernel',
        'matplotlib',
        'numpy',
    ],
)
