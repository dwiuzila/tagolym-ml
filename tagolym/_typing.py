from argparse import Namespace
from collections.abc import Iterable, Iterator
from json.encoder import JSONEncoder
from os import PathLike
from typing import Any, Literal, Optional, Union

import numpy as np
from numpy import ndarray
from optuna.trial._trial import Trial
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

FilePath = Union[str, "PathLike[str]"]
Transformer = Union[BaseEstimator, TransformerMixin]
RandomState = Union[int, ndarray, np.random.RandomState]
