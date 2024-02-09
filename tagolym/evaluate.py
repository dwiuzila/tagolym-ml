"""Given true labels and model predictions, the purpose of this module is to
calculate the precision, recall, f1 score of the model, and number of samples.
The performance is computed on the overall samples, per-class samples, and
per-slice samples. There are 8 slices considered:

- [X] short tokens, i.e. those that have less than 5 words,
- [X] six slices in which the posts are tagged as a subtopic but not tagged as
the bigger topic covering the subtopic, and
- [X] tokens that don't have frequent words with more than 3 letters.
"""

from sklearn.metrics import precision_recall_fscore_support
from snorkel.slicing import PandasSFApplier, SlicingFunction, slicing_function

from tagolym._typing import (
    DataFrame,
    Literal,
    Optional,
    Series,
    Union,
    ndarray,
)


@slicing_function()
def short_post(x: Series) -> bool:
    """Confirm whether a data point has a token with less than 5 words.

    Args:
        x (Series): Data point containing a token.

    Returns:
        Whether the data point has a token with less than 5 words.
    """
    return len(x["token"].split()) < 5


@slicing_function()
def inequality_not_algebra(x: Series) -> bool:
    """Confirm whether a data point has `"inequality"` but not `"algebra"` as
    one of its labels.

    Args:
        x (Series): Data point containing a list of labels.

    Returns:
        Whether the data point has `"inequality"` but not `"algebra"` as one
            of its labels.
    """
    inequality = "inequality" in x["tags"]
    algebra = "algebra" in x["tags"]
    return inequality and not algebra


@slicing_function()
def function_not_algebra(x: Series) -> bool:
    """Confirm whether a data point has `"function"` but not `"algebra"` as
    one of its labels.

    Args:
        x (Series): Data point containing a list of labels.

    Returns:
        Whether the data point has `"function"` but not `"algebra"` as one of
            its labels.
    """
    function = "function" in x["tags"]
    algebra = "algebra" in x["tags"]
    return function and not algebra


@slicing_function()
def polynomial_not_algebra(x: Series) -> bool:
    """Confirm whether a data point has `"polynomial"` but not `"algebra"` as
    one of its labels.

    Args:
        x (Series): Data point containing a list of labels.

    Returns:
        Whether the data point has `"polynomial"` but not `"algebra"` as one
            of its labels.
    """
    polynomial = "polynomial" in x["tags"]
    algebra = "algebra" in x["tags"]
    return polynomial and not algebra


@slicing_function()
def circle_not_geometry(x: Series) -> bool:
    """Confirm whether a data point has `"circle"` but not `"geometry"` as one
    of its labels.

    Args:
        x (Series): Data point containing a list of labels.

    Returns:
        Whether the data point has `"circle"` but not `"geometry"` as one of
            its labels.
    """
    circle = "circle" in x["tags"]
    geometry = "geometry" in x["tags"]
    return circle and not geometry


@slicing_function()
def trigonometry_not_geometry(x: Series) -> bool:
    """Confirm whether a data point has `"trigonometry"` but not `"geometry"`
    as one of its labels.

    Args:
        x (Series): Data point containing a list of labels.

    Returns:
        Whether the data point has `"trigonometry"` but not `"geometry"` as
            one of its labels.
    """
    trigonometry = "trigonometry" in x["tags"]
    geometry = "geometry" in x["tags"]
    return trigonometry and not geometry


@slicing_function()
def modular_arithmetic_not_number_theory(x: Series) -> bool:
    """Confirm whether a data point has `"modular arithmetic"` but not
    `"number theory"` as one of its labels.

    Args:
        x (Series): Data point containing a list of labels.

    Returns:
        Whether the data point has `"modular arithmetic"` but not `"number
            theory"` as one of its labels.
    """
    modular_arithmetic = "modular arithmetic" in x["tags"]
    number_theory = "number theory" in x["tags"]
    return modular_arithmetic and not number_theory


def keyword_lookup(x: Series, keywords: list) -> bool:
    """Confirm whether a token of a data point doesn't have frequent words
    with more than 3 characters.

    Args:
        x (Series): Data point containing a token.
        keywords (list): Frequent four-letter-or-more words derived from all
            tokens.

    Returns:
        Whether the token of the data point doesn't have frequent words with
            more than 3 letters.
    """
    return all(word not in x["token"].split() for word in keywords)


def make_keyword_sf(df: DataFrame) -> SlicingFunction:
    """Create a `SlicingFunction` object to use the [keyword_lookup]
    [evaluate.keyword_lookup] function.

    Args:
        df (DataFrame): Preprocessed data containing tokens and their
            corresponding labels.

    Returns:
        Python class for slicing functions, i.e. functions that take a data
            point as input and produce a boolean that states whether or not
            the data point satisfies some predefined conditions.
    """
    frequent_words = (
        df["token"].str.split(expand=True).stack().value_counts().index[:20]
    )
    keywords = [word for word in frequent_words if len(word) > 3]
    return SlicingFunction(
        name="without_frequent_words",
        f=keyword_lookup,
        resources=dict(keywords=keywords),
    )


def average_performance(
    y_true: ndarray,
    y_pred: ndarray,
    average: Optional[Literal["micro", "macro", "weighted"]] = "weighted",
) -> dict[str, Union[float, int]]:
    """Compute precision, recall, F-measure, and number of samples from model
    predictions and true labels.

    Args:
        y_true (ndarray): Ground truth (correct) target values.
        y_pred (ndarray): Estimated targets as returned by the model.
        average (Optional[Literal], optional): If `None`, the scores for each
            class are returned. Otherwise, this determines the type of
            averaging performed on the data:

            | Average      | Description                                      |
            | ------------ | ------------------------------------------------ |
            | `"micro"`    | Calculate metrics globally by counting the total \
                             true positives, false negatives and false        \
                             positives.                                       |
            | `"macro"`    | Calculate metrics for each label, and find their \
                             unweighted mean. This does not take label        \
                             imbalance into account.                          |
            | `"weighted"` | Calculate metrics for each label, and find their \
                             average weighted by support (the number of true  \
                             instances for each label). This alters `"macro"` \
                             to account for label imbalance; it can result in \
                             an F-score that is not between precision and     \
                             recall.                                          |

            Defaults to `"weighted"`.

    Returns:
        Dictionary containing precision, recall, F-measure, and number of
            samples.
    """
    metrics = precision_recall_fscore_support(y_true, y_pred, average=average)
    return {
        "precision": metrics[0],
        "recall": metrics[1],
        "f1": metrics[2],
        "num_samples": len(y_true),
    }


def get_slice_metrics(
    y_true: ndarray, y_pred: ndarray, slices: ndarray
) -> dict[str, dict]:
    """Apply [average_performance][evaluate.average_performance] with
    `"micro"` average to different slices of data.

    Args:
        y_true (ndarray): Ground truth (correct) target values.
        y_pred (ndarray): Estimated targets as returned by the model.
        slices (ndarray): Slices of data defined by slicing functions.

    Returns:
        Dictionary containing dictionaries of average performances across
            slices.
    """
    slice_metrics = {}
    for slice_name in slices.dtype.names:
        mask = slices[slice_name].astype(bool)
        if sum(mask):
            slice_metrics[slice_name] = average_performance(
                y_true[mask], y_pred[mask], "micro"
            )

    return slice_metrics


def get_metrics(
    y_true: ndarray, y_pred: ndarray, classes: ndarray, df: Optional[DataFrame] = None
) -> dict[str, dict]:
    """Compute model performance for the overall data (using "weighted"
    average), across classes, and across slices (using "micro" average).

    Args:
        y_true (ndarray): Ground truth (correct) target values.
        y_pred (ndarray): Estimated targets as returned by the model.
        classes (ndarray): Complete labels.
        df (Optional[DataFrame], optional): Preprocessed data containing
            tokens and their corresponding labels. Defaults to None.

    Returns:
        Dictionary containing dictionaries of average performances for the
            overall data, across classes, and across slices.
    """
    # performance
    performance = {"overall": {}, "class": {}}

    # overall performance
    performance["overall"] = average_performance(y_true, y_pred, "weighted")

    # per-class performance
    metrics = precision_recall_fscore_support(y_true, y_pred, average=None)
    for i in range(len(classes)):
        performance["class"][classes[i]] = {
            "precision": metrics[0][i],
            "recall": metrics[1][i],
            "f1": metrics[2][i],
            "num_samples": metrics[3][i],
        }

    # per-slice performance
    if df is not None:
        slices = PandasSFApplier(
            [
                short_post,
                inequality_not_algebra,
                function_not_algebra,
                polynomial_not_algebra,
                circle_not_geometry,
                trigonometry_not_geometry,
                modular_arithmetic_not_number_theory,
                make_keyword_sf(df),
            ]
        ).apply(df)
        performance["slices"] = get_slice_metrics(y_true, y_pred, slices)

    return performance
