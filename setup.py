# need to be in main file : pip install -e .

from setuptools import setup, find_packages
from setuptools import setup

package_name = "project"

setup(
    name=package_name,
    version="0.0.0",
    zip_safe=True,
    description="Motion Learning wiht Obstacle Avoidance",
    package_dir={"": "fall2022proj"},
    packages=find_packages(where="fall2022proj", include=["data", "src"]),
    # tests_require=["pytest"],
    # install_requires=["setuptools"],
    zip_safe=True,
    license="TODO",
)
