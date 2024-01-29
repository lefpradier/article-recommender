import numpy as np
import pandas as pd
import projet9_functions as p9
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf
import pickle as pkl
from termcolor import colored
import pickle as pkl
from implicit.bpr import BayesianPersonalizedRanking
from implicit.evaluation import mean_average_precision_at_k
from sklearn.model_selection import train_test_split


@hydra.main(version_base=None, config_path="../config", config_name="bpr")
def makerun(cfg: DictConfig):
    # pass user_config as sys.arg to merge config files
    if cfg.user_config is not None:
        user_config = OmegaConf.load(cfg.user_config)
        cfg = OmegaConf.merge(cfg, user_config)
    mlflow.set_tracking_uri("sqlite:///log.db")
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name=cfg.mlflow.run_name) as run:
        #!importation des data
        data = pd.read_csv(cfg.data)

        #!mise en forme
        # save usr item unique
        items = data["click_article_id"].unique().tolist()
        items.sort()
        usrs = data["user_id"].unique().tolist()
        usrs.sort()
        reads = {"usrs": usrs, "items": items}
        pkl.dump(reads, open("model/bpr_reads.pkl", "wb"))
        clicks_freq = p9.scoring(data)
        train, test = train_test_split(clicks_freq, random_state=42, test_size=0.2)
        train = p9.convert_to_spm(train, users=usrs, items=items)
        test = p9.convert_to_spm(test, users=usrs, items=items)
        print(colored("model", "blue"))
        #!model
        model = BayesianPersonalizedRanking(
            factors=cfg.n_factor,
            iterations=cfg.n_epoch,
            learning_rate=cfg.learning_rate,
            num_threads=20,
        )
        model.fit(train)
        with open("model/bpr.pkl", "wb") as fle:
            pkl.dump(model, fle, protocol=pkl.HIGHEST_PROTOCOL)

        #!params
        params = {
            "model_type": cfg.model_name,
            "n_epoch": cfg.n_epoch,
            "n_factors": cfg.n_factor,
            "learning_rate": cfg.learning_rate,
        }
        mlflow.log_params(params)

        #!scores
        print(colored("compute score", "red"))
        mapatk = mean_average_precision_at_k(model, train, test, K=5, num_threads=20)
        # dictionnaire de scores
        scores = {"mapatk": mapatk}
        print("mapatk: ", mapatk)
        #!artefact

        #!logging
        mlflow.log_metrics(scores)
        mlflow.log_artifact("model/bpr.pkl")
        mlflow.log_artifact("model/bpr_reads.pkl")


if __name__ == "__main__":
    makerun()
