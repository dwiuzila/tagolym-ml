import json
import joblib
import mlflow
import optuna
import tempfile
import pandas as pd
from pathlib import Path
from argparse import Namespace
from google.cloud import bigquery
from optuna.samplers import TPESampler
from google.oauth2 import service_account
from optuna.integration.mlflow import MLflowCallback

from config import config
from tagolym import predict, train, utils


def elt_data(key_path):
    # initialize bigquery client
    credentials = service_account.Credentials.from_service_account_file(
        key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    client = bigquery.Client(credentials=credentials, project=credentials.project_id,)

    # write query prompt
    query = """
    SELECT 
        post_canonical, 
        ANY_VALUE(ARRAY(
            SELECT * FROM UNNEST(tags)
        )) AS tags
    FROM `tag-math-olympiad.contest_collections.*`
    WHERE 
        category_name = "High School Olympiads"
        AND ARRAY_LENGTH(tags) > 0
    GROUP BY 1
    """

    # query to JSON format
    query_job = client.query(query)
    records = [dict(row) for row in query_job]

    # save data to local
    projects_fp = Path(config.DATA_DIR, "labeled_data.json")
    with open(projects_fp, "w") as fp:
        json.dump(records, fp)
    
    print("✅ Saved data!")
    

def train_model(args_fp, experiment_name, run_name):
    # load labeled data
    projects_fp = Path(config.DATA_DIR, "labeled_data.json")
    df = pd.read_json(projects_fp)

    # train
    args = Namespace(**utils.load_dict(filepath=args_fp))
    mlflow.set_experiment(experiment_name=experiment_name)

    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        print(f"Run ID: {run_id}")

        # fit, predict, and evaluate
        artifacts = train.train(args=args, df=df)

        # log key metrics
        for split in ["train", "val", "test"]:
            metrics = artifacts[f"{split}_metrics"]["overall"]
            for score in ["precision", "recall", "f1"]:
                mlflow.log_metrics({f"{split}_{score}": metrics[f"{score}"]})

        # log artifacts
        with tempfile.TemporaryDirectory() as dp:
            joblib.dump(artifacts["label_encoder"], Path(dp, "label_encoder.pkl"))
            joblib.dump(artifacts["model"], Path(dp, "model.pkl"))
            utils.save_dict(artifacts["train_metrics"], Path(dp, "train_metrics.json"), cls=utils.NumpyEncoder)
            utils.save_dict(artifacts["val_metrics"], Path(dp, "val_metrics.json"), cls=utils.NumpyEncoder)
            utils.save_dict(artifacts["test_metrics"], Path(dp, "test_metrics.json"), cls=utils.NumpyEncoder)
            utils.save_dict({**args.__dict__}, Path(dp, "args.json"), cls=utils.NumpyEncoder)
            mlflow.log_artifacts(dp)

        # log parameters
        mlflow.log_params(vars(artifacts["args"]))

    # save to config
    open(Path(config.CONFIG_DIR, "run_id.txt"), "w").write(run_id)
    utils.save_dict(
        artifacts["train_metrics"],
        Path(config.CONFIG_DIR, "train_metrics.json"),
        cls=utils.NumpyEncoder,
    )
    utils.save_dict(
        artifacts["val_metrics"],
        Path(config.CONFIG_DIR, "val_metrics.json"),
        cls=utils.NumpyEncoder,
    )
    utils.save_dict(
        artifacts["test_metrics"],
        Path(config.CONFIG_DIR, "test_metrics.json"),
        cls=utils.NumpyEncoder,
    )


def optimize(args_fp, study_name, num_trials):
    # load labeled data
    projects_fp = Path(config.DATA_DIR, "labeled_data.json")
    df = pd.read_json(projects_fp)
    args = Namespace(**utils.load_dict(filepath=args_fp))

    # define mlflow callback
    mlflow_callback = MLflowCallback(tracking_uri=mlflow.get_tracking_uri(), metric_name="f1")

    # optimize some args
    study = optuna.create_study(
        sampler=TPESampler(seed=config.SEED), study_name=study_name, direction="maximize"
    )
    study.optimize(
        lambda trial: train.objective(args, df, trial, experiment=0),
        n_trials=num_trials,
        callbacks=[mlflow_callback],
    )

    # update args
    curr_best_value = study.best_value
    args = {**args.__dict__, **study.best_params}
    args = Namespace(**args)

    # optimize other args
    study = optuna.create_study(
        sampler=TPESampler(seed=config.SEED), study_name=study_name, direction="maximize"
    )
    study.optimize(
        lambda trial: train.objective(args, df, trial, experiment=1),
        n_trials=num_trials,
        callbacks=[mlflow_callback],
    )

    # best trial
    if study.best_value > curr_best_value:
        args = {**args.__dict__, **study.best_params}
    else:
        args = args.__dict__
    
    # save to config
    utils.save_dict(args, Path(config.CONFIG_DIR, "args_opt.json"), cls=utils.NumpyEncoder)
    print(f"Best value (f1): {study.best_value}")
    print(f"Best hyperparameters: {json.dumps(args, indent=2)}")


def load_artifacts(run_id=None):
    # get run id
    if not run_id:
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()

    # locate specific artifacts directory
    experiment_id = mlflow.get_run(run_id=run_id).info.experiment_id
    artifacts_dir = Path(config.MODEL_REGISTRY, experiment_id, run_id, "artifacts")
    
    # load objects from run
    mlb = joblib.load(Path(artifacts_dir, "label_encoder.pkl"))
    model = joblib.load(Path(artifacts_dir, "model.pkl"))
    train_metrics = utils.load_dict(filepath=Path(artifacts_dir, "train_metrics.json"))
    val_metrics = utils.load_dict(filepath=Path(artifacts_dir, "val_metrics.json"))
    test_metrics = utils.load_dict(filepath=Path(artifacts_dir, "test_metrics.json"))
    args = Namespace(**utils.load_dict(filepath=Path(artifacts_dir, "args.json")))

    return {
        "args": args,
        "label_encoder": mlb,
        "model": model,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def predict_tag(text, run_id=None):
    # get run id
    if not run_id:
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    
    # load artifacts and predict
    artifacts = load_artifacts(run_id=run_id)
    prediction = predict.predict(texts=text, artifacts=artifacts)
    print(json.dumps(prediction, indent=2))
    return prediction