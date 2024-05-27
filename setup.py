from setuptools import find_packages, setup

with open("requirements.txt", "r") as f:
    requirements = f.readlines()
    
setup(
    name="All-Day-Vehicle-Detection",
    version="1.0.0", 
    description="This repository is used for detecting vehicles both during the day and at night",
    packages=find_packages(),
    install_requires = requirements, 
    classifiers = "Programming Language :: Python :: 3"
)