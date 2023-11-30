import numpy as np
import pandas as pd
import projet9_functions as p9
import projet9_recommenders as reco

import mlflow
import hydra
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig, OmegaConf
import pickle as pkl
from termcolor import colored
from implicit.bpr import BayesianPersonalizedRanking


@hydra.main(version_base=None, config_path="../config", config_name="hybrid")
def makerun(cfg: DictConfig):
    # pass user_config as sys.arg to merge config files
    if cfg.user_config is not None:
        user_config = OmegaConf.load(cfg.user_config)
        cfg = OmegaConf.merge(cfg, user_config)
    mlflow.set_tracking_uri("sqlite:///log.db")
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name=cfg.mlflow.run_name) as run:
        data = pd.read_csv(cfg.data)

        #!mide en forme
        # save usr item unique
        items = data["click_article_id"].unique().tolist()
        items.sort()
        usrs = data["user_id"].unique().tolist()
        usrs.sort()
        reads = {"usrs": usrs, "items": items}
        clicks_freq = p9.scoring(data)
        train, test = train_test_split(clicks_freq, random_state=42, test_size=0.2)
        train_tmp = p9.convert_to_spm(train, users=usrs, items=items)
        test = p9.convert_to_spm(test, users=usrs, items=items)
        test_df = pd.DataFrame(
            {"user_id": test.tocoo().row, "article_id": test.tocoo().col}
        )
        relevant = p9.relevant(test_df)

        #!model
        model_cf = BayesianPersonalizedRanking(
            factors=int(cfg.n_factor),
            iterations=cfg.n_epoch,
            learning_rate=cfg.learning_rate,
            num_threads=20,
        )
        model_cf.fit(train_tmp)
        with open("model/bpr%s.pkl" % str(cfg.n_factor), "wb") as fle:
            pkl.dump(model_cf, fle, protocol=pkl.HIGHEST_PROTOCOL)
        model_pop = pd.read_csv("model/popularity.csv")
        model_cb = pd.read_csv("model/content_similarity.csv")
        model = reco.hybrid_recommender_bpr(
            model_cf, model_pop, model_cb, reads, int(cfg.thr_pop), cfg.thr_cb
        )
        top5 = model.recommend(train)

        #!params
        params = {
            "model_type": cfg.model_name,
            "thr_pop": cfg.thr_pop,
            "thr_cb": cfg.thr_cb,
            "n_epoch": cfg.n_epoch,
            "n_factor": cfg.n_factor,
            "learning_rate": cfg.learning_rate,
        }
        mlflow.log_params(params)

        #!scores
        mapatk = p9.map_at_k(relevant, top5)
        # dictionnaire de scores
        scores = {"mapatk": mapatk}

        #!artefact
        fig, perf = p9.plot_apatk(relevant, top5)
        fig.savefig("plots/apatk_%s_%s.png" % (cfg.model_name, cfg.max_sampled))
        perf.to_csv(
            "performances/apatk_relevant_%s_%s" % (cfg.model_name, cfg.max_sampled),
            index=False,
        )

        mlflow.log_metrics(scores)
        mlflow.log_artifact("plots/apatk_%s_%s.png" % (cfg.model_name, cfg.max_sampled))
        mlflow.log_artifact(
            "performances/apatk_relevant_%s_%s" % (cfg.model_name, cfg.max_sampled)
        )


if __name__ == "__main__":
    makerun()
