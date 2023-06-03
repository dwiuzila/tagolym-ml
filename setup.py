from pathlib import Path
from setuptools import find_namespace_packages, setup

# load libraries from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt")) as file:
    required_packages = [ln.strip() for ln in file.readlines()]

# define the package
setup(
    name="tagolym",
    version=0.1,
    description="Classify math olympiad problems.",
    author="Albers Uzila",
    author_email="tagolym@gmail.com",
    url="https://dwiuzila.medium.com/",
    python_requires=">=3.7",
    packages=find_namespace_packages(),
    install_requires=[required_packages],
)