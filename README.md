# tagolym-ml

[![Built with Material for MkDocs](https://img.shields.io/badge/Material_for_MkDocs-526CFE?style=for-the-badge&logo=MaterialForMkDocs&logoColor=white)](https://squidfunk.github.io/mkdocs-material/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Tag high school math olympiad problems with 10 predefined topics:

| Big Topics         | Algebra Subtopics  | Geometry Subtopics | Number Theory Subtopics |
|--------------------|--------------------|--------------------|-------------------------|
| algebra            | inequality         | circle             | modular arithmetic      |
| geometry           | function           | trigonometry       |                         |
| number theory      | polynomial         |                    |                         |
| combinatorics      |                    |                    |                         |

**Input text:**
> Find all functions $f:(0,\infty)\rightarrow (0,\infty)$ such that for any $x,y\in (0,\infty)$, 
> 
> $$xf(x^2)f(f(y)) + f(yf(x)) = f(xy) \left(f(f(x^2)) + f(f(y^2))\right).$$

**Predicted tags:**
> ["algebra", "function"]

## Virtual Environment
```console
$ git clone https://github.com/dwiuzila/tagolym-ml.git
$ cd tagolym-ml
$ git checkout style
$ make venv
```

## Directory
```
config/
├── args_opt.json         - optimized parameters
├── args.json             - preprocessing/training parameters
├── config.py             - configuration setup
├── run_id.txt            - run id of the last model training
├── test_metrics.json     - model performance on test split
├── train_metrics.json    - model performance on train split
└── val_metrics.json      - model performance on validation split

docs/
├── tagolym/
│   ├── data.md           - documentation for data.py
│   ├── evaluate.md       - documentation for evaluate.py
│   ├── main.md           - documentation for main.py
│   ├── predict.md        - documentation for predict.py
│   ├── train.md          - documentation for train.py
│   └── utils.md          - documentation for utils.py
├── index.md              - homepage
├── license.md            - project license
└── logo.png              - project logo

tagolym/
├── _typing.py            - type hints
├── data.py               - data processing components
├── evaluate.py           - evaluation components
├── main.py               - training/optimization pipelines
├── predict.py            - inference components
├── train.py              - training components
└── utils.py              - supplementary utilities

.flake8                   - code quality assurance

.gitignore                - files/folders that git will ignore

LICENSE                   - project license

Makefile                  - task automation

mkdocs.yml                - configuration file for docs

pyproject.toml            - build system dependencies

README.md                 - longform description of the project

requirements.txt          - package dependencies

setup.py                  - code packaging
```

## Workflow
You wouldn't be able to execute the `# query data` part in the code snippet below due to data access restrictions. For that, you'd need my credential, which unfortunately is not to be shared. But worry not, I'll provide samples for you to work with. What you need to do is simply [download](https://gist.github.com/dwiuzila/74dc99fe6f6d3901dbd1695f77977865) the samples `labeled_data.json` and save the file in a folder named `data` in the working directory.

```python
from pathlib import Path
from config import config
from tagolym import main

# query data
key_path = "credentials/bigquery-key.json"
main.elt_data(key_path)

# optimize model
args_fp = Path(config.CONFIG_DIR, "args.json")
main.optimize(args_fp, study_name="optimization", num_trials=10)

# train model
args_fp = Path(config.CONFIG_DIR, "args_opt.json")
main.train_model(args_fp, experiment_name="baselines", run_name="sgd")

# inference
texts = [
    "Let $c,d \geq 2$ be naturals. Let $\{a_n\}$ be the sequence satisfying $a_1 = c, a_{n+1} = a_n^d + c$ for $n = 1,2,\cdots$.Prove that for any $n \geq 2$, there exists a prime number $p$ such that $p|a_n$ and $p \not | a_i$ for $i = 1,2,\cdots n-1$.",
    "Let $ABC$ be a triangle with circumcircle $\Gamma$ and incenter $I$ and let $M$ be the midpoint of $\overline{BC}$. The points $D$, $E$, $F$ are selected on sides $\overline{BC}$, $\overline{CA}$, $\overline{AB}$ such that $\overline{ID} \perp \overline{BC}$, $\overline{IE}\perp \overline{AI}$, and $\overline{IF}\perp \overline{AI}$. Suppose that the circumcircle of $\triangle AEF$ intersects $\Gamma$ at a point $X$ other than $A$. Prove that lines $XD$ and $AM$ meet on $\Gamma$.",
    "Find all functions $f:(0,\infty)\rightarrow (0,\infty)$ such that for any $x,y\in (0,\infty)$, $$xf(x^2)f(f(y)) + f(yf(x)) = f(xy) \left(f(f(x^2)) + f(f(y^2))\right).$$",
    "Let $n$ be an even positive integer. We say that two different cells of a $n \times n$ board are [b]neighboring[/b] if they have a common side. Find the minimal number of cells on the $n \times n$ board that must be marked so that any cell (marked or not marked) has a marked neighboring cell."
]
main.predict_tag(texts=texts)
```

## Documentation
Open full documentation: https://dwiuzila.github.io/tagolym-ml/.

```console
$ git checkout documentation
$ pip install -e ".[docs]"
$ mkdocs gh-deploy --force
```

## Makefile

```console
$ make help
Commands:
venv    : creates a virtual environment.
style   : executes style formatting.
clean   : cleans all unnecessary files.
```