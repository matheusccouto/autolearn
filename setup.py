from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Get requirements
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='autolearn',
    version='0.1.0',
    description='Automated machine learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/matheusccouto/autolearn',
    author='Matheus Couto',
    author_email='matheusccouto@gmail.com',
    packages=['autolearn'],
    python_requires='>=3.6',
    install_requires=required,
)
