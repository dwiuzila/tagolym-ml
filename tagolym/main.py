"""The main module that runs everything end-to-end:

1. Extract, load, and transform raw data
2. Utilize preprocessed data to optimize the pipeline 
3. Train a model with the best arguments
4. Predict on new data using the trained model
"""

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
from tagolym._typing import Any, Optional, FilePath

from config import config
from config.config import logger
from tagolym import predict, train, utils


def elt_data(key_path: FilePath) -> None:
    """Query raw data from BigQuery and save it to `data` folder in JSON 
    format.

    Args:
        key_path (FilePath): Path to the Google service account private key 
            JSON file used for creating credentials.
    """    
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
    
    logger.info("✅ Saved data!")
    

def train_model(args_fp: FilePath, experiment_name: str, run_name: str) -> None:
    """Load raw data from `data` folder and pass it to [train.train][] to 
    preprocess the data and train a model with it. Log the metrics, artifacts, 
    and parameters using MLflow. Save also MLflow run ID and metrics to 
    `config` folder.

    Args:
        args_fp (FilePath): Path to the arguments used. Arguments include 
            booleans for preprocessing the posts (whether to exclude command 
            words and to implement word stemming), hyperparameters for the 
            modeling pipeline, and the best threshold for each class.
        experiment_name (str): User input experiment name for MLflow.
        run_name (str): User input run name for MLflow.
    """
    # load labeled data
    projects_fp = Path(config.DATA_DIR, "labeled_data.json")
    df = pd.read_json(projects_fp)

    # train
    args = Namespace(**utils.load_dict(filepath=args_fp))
    mlflow.set_experiment(experiment_name=experiment_name)

    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        logger.info(f"Run ID: {run_id}")

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


def optimize(args_fp: FilePath, study_name: str, num_trials: int) -> None:
    """Load raw data from `data` folder and optimize given arguments by 
    maximizing the f1 score in validation split. For search efficiency, the 
    optimization is done in two steps:

    1. for hyperparameters in preprocessing, vectorization, and modeling; and 
    2. for hyperparameters in the learning algorithm.
    
    Save also the best arguments to `config` folder, name it as 
    `args_opt.json`.

    Args:
        args_fp (FilePath): Path to the initial arguments for the entire 
            process. Arguments include booleans for preprocessing the posts and 
            hyperparameters for the modeling pipeline.
        study_name (str): User input study name for MLflow.
        num_trials (int): Number of trials for arguments tuning, at minimum 1.
    """
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
    utils.save_dict(dict(args), Path(config.CONFIG_DIR, "args_opt.json"), cls=utils.NumpyEncoder)
    logger.info(f"Best value (f1): {study.best_value}")
    logger.info(f"Best hyperparameters: {json.dumps(args, indent=2)}")


def load_artifacts(run_id: Optional[str] = None) -> dict[str, Any]:
    """Load the artifacts of a specific MLflow run ID into memory, including 
    arguments, metrics, model, and label binarizer.

    Args:
        run_id (Optional[str], optional): MLflow run ID. Defaults to None.

    Returns:
        Final artifacts used for inference.
    """
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


def predict_tag(texts: list[str], run_id: Optional[str] = None) -> list[dict]:
    """Given a specific MLflow run ID and some posts, predict their labels 
    using preloaded artifacts by calling [predict.predict][].

    Args:
        texts (list[str]): List of posts.
        run_id (Optional[str], optional): MLflow run ID. If None, run ID will 
            be set from `run_id.txt` inside `config` folder. Defaults to None.

    Returns:
        List of key-value pairs of post and its label prediction.
    """
    # get run id
    if not run_id:
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    
    # load artifacts and predict
    artifacts = load_artifacts(run_id=run_id)
    prediction = predict.predict(texts=texts, artifacts=artifacts)
    logger.info(json.dumps(prediction, indent=2))
    return prediction