import numpy as np
from tagolym import data, train


def custom_predict(X, model, args, y_true=None):
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)
        y_score = np.array(y_score)[:, :, 1].T
        if y_true is not None:
            args.threshold = train.tune_threshold(y_true, y_score)
        y_pred = y_score > args.threshold
    else:
        y_pred = model.predict(X)
    return y_pred, args


def predict(texts, artifacts):
    # load artifacts
    args = artifacts["args"]
    mlb = artifacts["label_encoder"]
    model = artifacts["model"]

    # predict
    x = [data.preprocess_post(txt, args.nocommand, args.stem) for txt in texts]
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