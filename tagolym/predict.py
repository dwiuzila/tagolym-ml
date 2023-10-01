"""Prediction module, called after model training."""

import numpy as np
import pandas as pd
from tagolym._typing import ndarray, Series, Pipeline, Any, Optional, Namespace
from tagolym import data, train


def custom_predict(X: Series, model: Pipeline, args: Namespace, y_true: Optional[ndarray] = None) -> tuple[ndarray, Namespace]:
    """If the model has `predict_proba` attribute, predict the probability of 
    each label occurring. Furthermore, if the true labels are given, use them 
    to tune the threshold for each class using [train.tune_threshold][]. 
    Otherwise, if the model has no `predict_proba` attribute, predict the 
    label directly (0 or 1) using 0.5 threshold.

    Args:
        X (Series): Preprocessed posts.
        model (Pipeline): End-to-end pipeline including vectorizer and model.
        args (Namespace): Arguments containing booleans for preprocessing the 
            posts and hyperparameters for the modeling pipeline. Can also 
            contain the best threshold tuned for each class.
        y_true (Optional[ndarray], optional): Ground truth (correct) target 
            values. Defaults to None.

    Returns:
        y_pred: Estimated targets as returned by the model.
        args: Arguments, either is the same as input arguments or additionally 
            also contains the best threshold tuned for each class.
    """
    # prioritize predict_proba over predict
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)
        y_score = np.array(y_score)[:, :, 1].T

        # tune threshold if label is given
        if y_true is not None:
            args.threshold = train.tune_threshold(y_true, y_score)
        
        y_pred = y_score > args.threshold
    else:
        y_pred = model.predict(X)
    return y_pred, args


def predict(texts: list[str], artifacts: dict[str, Any]) -> list[dict]:
    """Load arguments, label binarizer, and the trained model. Then, 
    preprocess given posts and predict their labels using 
    [custom_predict][predict.custom_predict]. The label binarizer is used to 
    transform the prediction matrix back into readable labels.

    Args:
        texts (list[str]): User input list of posts.
        artifacts (dict[str, Any]): Arguments, label binarizer, and the 
            trained model.

    Returns:
        List of key-value pairs of post and its label prediction.
    """
    # load artifacts
    args = artifacts["args"]
    mlb = artifacts["label_encoder"]
    model = artifacts["model"]

    # predict
    x = pd.Series([data.preprocess_post(txt, args.nocommand, args.stem) for txt in texts])
    y, args = custom_predict(x, model, args)
    tags = mlb.inverse_transform(y)
    predictions = [
        {
            "input_text": texts[i],
            "predicted_tags": tags[i],
        }
        for i in range(len(texts))
    ]
    return predictions