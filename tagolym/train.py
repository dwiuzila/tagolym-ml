import numpy as np
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from config import config
from tagolym import data, evaluate, predict


def train(args, df):
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


def objective(args, df, trial, experiment=0):
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


def tune_threshold(y_true, y_score):
    # initialize threshold grid
    grid = np.linspace(0, 1, 101)
    threshold = []
    
    # find best threshold for each class
    for yt, ys in zip(y_true.T, y_score.T):
        f1 = {}
        for th in grid:
            yp = (ys > th).astype(int)
            f1[th] = f1_score(yt, yp)
        best_th = max(f1, key=f1.get)
        threshold.append(best_th)
    
    return threshold