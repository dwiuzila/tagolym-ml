from sklearn.metrics import precision_recall_fscore_support
from snorkel.slicing import PandasSFApplier, SlicingFunction, slicing_function


@slicing_function()
def short_post(x):
    return len(x["token"].split()) < 5


@slicing_function()
def inequality_not_algebra(x):
    inequality = "inequality" in x["tags"]
    algebra = "algebra" in x["tags"]
    return (inequality and not algebra)


@slicing_function()
def function_not_algebra(x):
    function = "function" in x["tags"]
    algebra = "algebra" in x["tags"]
    return (function and not algebra)


@slicing_function()
def polynomial_not_algebra(x):
    polynomial = "polynomial" in x["tags"]
    algebra = "algebra" in x["tags"]
    return (polynomial and not algebra)


@slicing_function()
def circle_not_geometry(x):
    circle = "circle" in x["tags"]
    geometry = "geometry" in x["tags"]
    return (circle and not geometry)


@slicing_function()
def trigonometry_not_geometry(x):
    trigonometry = "trigonometry" in x["tags"]
    geometry = "geometry" in x["tags"]
    return (trigonometry and not geometry)


@slicing_function()
def modular_arithmetic_not_number_theory(x):
    modular_arithmetic = "modular arithmetic" in x["tags"]
    number_theory = "number theory" in x["tags"]
    return (modular_arithmetic and not number_theory)


def keyword_lookup(x, keywords):
    return all(word not in x["token"].split() for word in keywords)


def make_keyword_sf(df):
    frequent_words = df["token"].str.split(expand=True).stack().value_counts().index[:20]
    keywords = [word for word in frequent_words if len(word) >= 4]
    return SlicingFunction(
        name="without_frequent_words",
        f=keyword_lookup,
        resources=dict(keywords=keywords),
    )


def average_performance(y_true, y_pred, average="weighted"):
    metrics = precision_recall_fscore_support(y_true, y_pred, average=average)
    return {
        "precision": metrics[0],
        "recall": metrics[1],
        "f1": metrics[2],
        "num_samples": len(y_true),
    }


def get_slice_metrics(y_true, y_pred, slices):
    slice_metrics = {}
    for slice_name in slices.dtype.names:
        mask = slices[slice_name].astype(bool)
        if sum(mask):
            slice_metrics[slice_name] = average_performance(y_true[mask], y_pred[mask], "micro")

    return slice_metrics


def get_metrics(y_true, y_pred, classes, df=None):
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
        slices = PandasSFApplier([
            short_post,
            inequality_not_algebra,
            function_not_algebra,
            polynomial_not_algebra,
            circle_not_geometry,
            trigonometry_not_geometry,
            modular_arithmetic_not_number_theory,
            make_keyword_sf(df),
        ]).apply(df)
        performance["slices"] = get_slice_metrics(y_true, y_pred, slices)

    return performance