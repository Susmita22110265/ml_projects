from setuptools import find_packages, setup
from typing import List

HYPHEN_DOT = '-e .'

def get_requirements(file_path :str )->List[str]:    
    '''
    this functon will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
     requirements = file_obj.readlines()
     requirements = [req.replace("\n", " ") for req in requirements]

    if HYPHEN_DOT in requirements : 
       requirements.remove(HYPHEN_DOT)

    return requirements


setup(
    name = "my_package",
    version = '0.0.1',
    author = 'Susmita R', 
    author_email = 'susmita.r@iitgn.ac.in',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
    )