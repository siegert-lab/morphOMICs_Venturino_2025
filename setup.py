from setuptools import setup, find_packages

setup(
    name="kxa_analysis",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},  # Root of the package is in the src directory
)
