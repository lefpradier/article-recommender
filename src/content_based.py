import numpy as np
import pandas as pd
import os
import projet9_functions as p9
import projet9_recommenders as reco
import mlflow
import hydra
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../config", config_name="content_based")
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
        clicks_freq = p9.scoring_tfidf(data)
        # clicks_freq["count"] = 1  # annule le score
        train, test = train_test_split(clicks_freq, test_size=0.2)
        train.rename(columns={"score": "freq"}, inplace=True)
        relevant = p9.relevant(test)

        #!model
        if os.path.exists("model/content_similarity.csv"):
            sim = pd.read_csv("model/content_similarity.csv")
        else:
            sim = p9.get_similarity(cfg.top_k, cfg.pca_n_comp, red=False)
            sim.to_csv("model/content_similarity.csv")
        model = reco.cb_recommender(sim)
        top5 = model.recommend(train)

        #!params
        params = {
            "model_type": cfg.model_name,
            "top_k": cfg.top_k,
            "PCA_ncomp": cfg.pca_n_comp,
        }
        mlflow.log_params(params)

        #!scores
        mapatk = p9.map_at_k(relevant, top5)
        # dictionnaire de scores
        scores = {"mapatk": mapatk}

        #!artefact
        fig, perf = p9.plot_apatk(relevant, top5)
        fig.savefig("plots/apatk_%s.png" % cfg.model_name)
        perf.to_csv("performances/apatk_relevant_%s" % cfg.model_name, index=False)

        mlflow.log_metrics(scores)
        mlflow.log_artifact("model/content_similarity.csv")
        mlflow.log_artifact("plots/apatk_%s.png" % cfg.model_name)
        mlflow.log_artifact("performances/apatk_relevant_%s" % cfg.model_name)


if __name__ == "__main__":
    makerun()
