from pathlib import Path
from setuptools import find_namespace_packages, setup

# load libraries from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt")) as file:
    required_packages = [ln.strip() for ln in file.readlines()]

type_packages = ["mypy==1.5.1"]

docs_packages = [
    "mkdocs-material==9.3.1",
    "mkdocstrings-python==1.7.0"
]

# define the package
setup(
    name="tagolym",
    version=0.2,
    description="Classify math olympiad problems.",
    author="Albers Uzila",
    author_email="tagolym@gmail.com",
    url="https://dwiuzila.github.io/tagolym-ml/",
    python_requires=">=3.9",
    packages=find_namespace_packages(),
    install_requires=[required_packages],
    extras_require={
        "dev": type_packages + docs_packages,
        "docs": docs_packages,
    },
)