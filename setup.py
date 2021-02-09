from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='auto-ml-models',
      version='0.0',
      description='Pre employment test',
      author='Aliaksandr Laskov',
      license='All rights reserved.',
      packages=find_packages(),
      zip_safe=False,
      install_requires=required,
      dependency_links=[]
      )
