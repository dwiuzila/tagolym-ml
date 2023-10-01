import numpy as np
from os import PathLike
from numpy import ndarray
from argparse import Namespace
from json.encoder import JSONEncoder
from sklearn.pipeline import Pipeline
from optuna.trial._trial import Trial
from pandas.core.series import Series
from pandas.core.frame import DataFrame
from collections.abc import Iterable, Iterator
from typing import Any, Literal, Optional, Union
from sklearn.base import BaseEstimator, TransformerMixin

FilePath = Union[str, "PathLike[str]"]
Transformer = Union[BaseEstimator, TransformerMixin]
RandomState = Union[int, ndarray, np.random.RandomState]