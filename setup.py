from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT = "-e ."
def get_requirements(dir:str) -> List[str]:
    """This function will list of requiremnts"""
    requirements = []
    with open(dir) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name = "mlproject",
    version = "0.0.1",
    author = "Prakash",
    author_email= "Prakash.mewari@yahoo.com",
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)