from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()


setup(
    name="unbound_models",
    version="0.0.0",
    description="",
    url="https://github.com/sshh12/unbound_models",
    author="Shrivu Shankar",
    license="Apache License 2.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=required,
)
