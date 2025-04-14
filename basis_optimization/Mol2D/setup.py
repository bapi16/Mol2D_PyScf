from setuptools import setup, find_packages

setup(
    name="Mol2D",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.6",
    ],
    python_requires=">=3.6",
)
