"""Training and optimization module, called after extracting, loading, and 
transforming raw data.
"""

import numpy as np
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from tagolym._typing import ndarray, DataFrame, Any, Namespace, Trial

from config import config
from tagolym import data, evaluate, predict


def train(args: Namespace, df: DataFrame) -> dict[str, Any]:
    """Preprocess the data, binarize the labels, and split the data using 
    functions from [data][] module. Then, initialize a model, train it, 
    predict the labels on all three splits using the trained model, and 
    evaluate the predictions. This function accepts arguments, to which an 
    additional argument `threshold` may be added before being returned. 
    Basically, `threshold` is a list of the best threshold tuned for each 
    class.

    Args:
        args (Namespace): Arguments containing booleans for preprocessing the 
            posts and hyperparameters for the modeling pipeline.
        df (DataFrame): Raw data containing posts and their corresponding tags.

    Returns:
        Artifacts containing arguments, label binarizer, and the trained model.
    """
    # setup
    df = data.preprocess(df, args.nocommand, args.stem)
    tags, mlb = data.binarize(df["tags"])
    classes = mlb.classes_
    X_train, X_val, X_test, y_train, y_val, y_test = data.split_data(
        df[["token", "tags"]], tags, random_state=config.SEED
    )

    # model
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, args.ngram_max))),
        ("multilabel", MultiOutputClassifier(
            SGDClassifier(
                penalty="elasticnet",
                random_state=config.SEED,
                early_stopping=True,
                class_weight="balanced",
                loss=args.loss,
                alpha=args.alpha,
                l1_ratio=args.l1_ratio,
                learning_rate=args.learning_rate,
                eta0=args.eta0,
                power_t=args.power_t,
            ),
            n_jobs=-1,
        )),
    ])
    
    # fit, predict, and evaluate
    model.fit(X_train["token"], y_train)
    
    y_pred, args = predict.custom_predict(X_val["token"], model, args, y_true=y_val)
    val_metrics = evaluate.get_metrics(y_val, y_pred, classes, df=X_val)
    
    y_pred, args = predict.custom_predict(X_train["token"], model, args)
    train_metrics = evaluate.get_metrics(y_train, y_pred, classes, df=X_train)

    y_pred, args = predict.custom_predict(X_test["token"], model, args)
    test_metrics = evaluate.get_metrics(y_test, y_pred, classes, df=X_test)

    return {
        "args": args,
        "label_encoder": mlb,
        "model": model,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def objective(args: Namespace, df: DataFrame, trial: Trial, experiment: int = 0) -> float:
    """F1 score is a metric chosen to be optimized in hyperparameter tuning. 
    Using arguments chosen in an optuna trial, this function trains the model 
    using [train][train.train] and returns the f1 score of the validation 
    split. It also sets additional attributes to the trial, including 
    precision, recall, and the f1 score on all three splits.

    Args:
        args (Namespace): Arguments containing booleans for preprocessing the 
            posts and hyperparameters for the modeling pipeline.
        df (DataFrame): Raw data containing posts and their corresponding tags.
        trial (Trial): Process of evaluating an objective function. This 
            object is passed to an objective function and provides interfaces 
            to get parameter suggestion, manage the trial's state, and set/get 
            user-defined attributes of the trial.
        experiment (int, optional): Index for two-step optimization: 
            optimizing hyperparameters in preprocessing, vectorization, and 
            modeling; and hyperparameters in the learning algorithm. Defaults 
            to 0.

    Raises:
        ValueError: Experiment index is neither 0 nor 1.

    Returns:
        F1 score of the validation split.
    """
    # parameters to tune
    if experiment == 0:
        args.nocommand = trial.suggest_categorical("nocommand", [True, False])
        args.stem = trial.suggest_categorical("stem", [True, False])
        args.ngram_max = trial.suggest_int("ngram_max", 2, 4)
        args.loss = trial.suggest_categorical("loss", ["hinge", "log_loss", "modified_huber"])
        args.l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
        args.alpha = trial.suggest_float("alpha", 1e-5, 1e-2, log=True)
    elif experiment == 1:
        args.learning_rate = trial.suggest_categorical(
            "learning_rate", ["constant", "optimal", "invscaling", "adaptive"]
        )
        if args.learning_rate != "optimal":
            args.eta0 = trial.suggest_float("eta0", 1e-2, 1e-0, log=True)
        if args.learning_rate == "invscaling":
            args.power_t = trial.suggest_float("power_t", 0.1, 0.5)
    else:
        raise ValueError("Experiment not recognized. Try 0 or 1.")

    # train
    artifacts = train(args=args, df=df)

    # set additional attributes
    for split in ["train", "val", "test"]:
        metrics = artifacts[f"{split}_metrics"]["overall"]
        for score in ["precision", "recall", "f1"]:
            trial.set_user_attr(f"{split}_{score}", metrics[f"{score}"])

    return artifacts["val_metrics"]["overall"]["f1"]


def tune_threshold(y_true: ndarray, y_score: ndarray) -> list:
    """The default decision boundary for a binary classification problem is 
    0.5, which may not be optimal depending on the problem. So, besides tuning 
    arguments, the threshold for each class is also tuned by optimizing the f1 
    score. What it does is try all possible values of the threshold in a grid 
    from 0 to 1 and pick the one that has the maximum f1 score.

    Args:
        y_true (ndarray): Ground truth (correct) target values.
        y_score (ndarray): Prediction probability of the model.

    Returns:
        List of the best threshold for each class.
    """
    # initialize threshold grid
    grid = np.linspace(0, 1, 101)
    threshold = []
    
    # find best threshold for each class
    for yt, ys in zip(y_true.T, y_score.T):
        f1 = {}
        for th in grid:
            yp = (ys > th).astype(int)
            f1[th] = f1_score(yt, yp)
        best_th = max(f1, key=f1.__getitem__)
        threshold.append(best_th)
    
    return threshold