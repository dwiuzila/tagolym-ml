from pathlib import Path

from setuptools import find_namespace_packages, setup

# load libraries from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt")) as file:
    required_packages = [ln.strip() for ln in file.readlines()]

docs_packages = ["mypy==1.5.1", "mkdocs-material==9.3.1", "mkdocstrings-python==1.7.0"]

style_packages = ["black==23.12.1", "flake8==7.0.0", "isort==5.13.2"]

# define the package
setup(
    name="tagolym",
    version=0.3,
    description="Classify math olympiad problems.",
    author="Albers Uzila",
    author_email="tagolym@gmail.com",
    url="https://dwiuzila.github.io/tagolym-ml/",
    python_requires=">=3.9",
    packages=find_namespace_packages(),
    install_requires=[required_packages],
    extras_require={
        "dev": docs_packages + style_packages,
        "docs": docs_packages,
    },
)
