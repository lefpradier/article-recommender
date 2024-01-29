import numpy as np
import pandas as pd
import projet9_functions as p9
import projet9_recommenders as reco
import mlflow
import hydra
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../config", config_name="popularity")
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

        #!Mise en forme
        clicks_freq = p9.scoring(data)
        train, test = train_test_split(clicks_freq, test_size=0.2)
        train.rename(columns={"count": "freq"}, inplace=True)
        relevant = p9.relevant(test)

        #!model
        pop = p9.get_popularity(data)
        pop.to_csv("model/popularity.csv", index=False)
        model = reco.pop_recommender(pop)
        top5 = model.recommend(train)

        #!params
        params = {"model_type": cfg.model_name}
        mlflow.log_params(params)

        #!scores
        mapatk = p9.map_at_k(relevant, top5)
        # dictionnaire de scores
        scores = {"mapatk": mapatk}

        #!artefact
        fig, perf = p9.plot_apatk(relevant, top5)
        fig.savefig("plots/apatk_%s.png" % cfg.model_name)
        perf.to_csv("performances/apatk_relevant_%s" % cfg.model_name, index=False)

        #!logging
        mlflow.log_metrics(scores)
        mlflow.log_artifact("model/popularity.csv")
        mlflow.log_artifact("plots/apatk_%s.png" % cfg.model_name)
        mlflow.log_artifact("performances/apatk_relevant_%s" % cfg.model_name)


if __name__ == "__main__":
    makerun()
