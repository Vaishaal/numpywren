# pylint: skip-file
#!/usr/bin/env python
import sys

#import pkgconfig
from setuptools import setup, find_packages

if sys.version_info < (3, 6):
    sys.exit('Sorry, Python < 3.6 is not supported')

# http://stackoverflow.com/questions/6344076/differences-between-distribute-distutils-setuptools-and-distutils2

# how to get version info into the project
exec(open('numpywren/version.py').read())

setup(
    name='numpywren',
    version=__version__,
    author='Vaishaal Shankar',
    description='Run scientific computing applications transparently on AWS Lambda and other cloud services',
    author_email='vaishaal@gmail.com',
    packages=find_packages(),
    install_requires=[
        'Click', 'PyYAML',
        'enum34', 'flaky', 'glob2',
        'watchtower', 'tblib', 'pywren' # it's nuts that we need both botos
    ],
    entry_points={
        'console_scripts' : ['numpywren=numpywren.scripts.cli:main']
    },
    tests_requires=[
        'pytest', 'numpy',
    ],
    include_package_data=True
)
