
from setuptools import find_packages
from setuptools import setup 

packages = find_packages()

setup(
    name                    = 'rl_algorithms',
    version                 = '0.0.1',
    description             = 'Implementations of various RL algorithms',
    author                  = 'Wilbert Santos Pumacay Huallpa',
    license                 = 'MIT License',
    author_email            = 'wpumacay@gmail.com',
    url                     = 'https://github.com/wpumacay/rl_algorithms',
    keywords                = 'rl ai dl',
    packages                = packages
)