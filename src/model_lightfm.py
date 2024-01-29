import numpy as np
import pandas as pd
import projet9_functions as p9
from tfrec.models import SVDpp, SVD
from tfrec.utils import preprocess_and_split
import mlflow
import hydra
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig, OmegaConf
import pickle as pkl
from termcolor import colored
import gc
from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
import pickle as pkl


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="collaborative_filtering_lightfm",
)
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
        popularity = pd.read_csv("model/popularity.csv")

        #!Mise en forme
        clicks_freq = p9.scoring(data)
        user_ids = clicks_freq["user_id"].unique().tolist()
        user_to_encoding = {user_ids[i]: i for i in range(len(user_ids))}
        item_ids = clicks_freq["article_id"].unique().tolist()
        item_to_encoding = {item_ids[i]: i for i in range(len(item_ids))}
        encoding_to_item = {i: item_ids[i] for i in range(len(item_ids))}
        encodings = (user_to_encoding, item_to_encoding, encoding_to_item)
        pkl.dump(encodings, open("model/lightfm_encodings.pkl", "wb"))
        clicks_freq_tn = p9.add_neg(clicks_freq, 300, popularity)
        clicks_freq = p9.convert_to_spm(clicks_freq)
        train, test = random_train_test_split(clicks_freq, random_state=42)
        #!Génération de négatifs
        clicks_freq_tn = clicks_freq_tn[clicks_freq_tn["count"] == 0]
        test = pd.DataFrame(
            {
                "user_id": test.tocoo().row,
                "article_id": test.tocoo().col,
                "count": test.tocoo().data,
            }
        )
        print(colored("convert", "blue"))
        clicks_freq_tn["user_id"] = clicks_freq_tn["user_id"].map(user_to_encoding)
        clicks_freq_tn["article_id"] = clicks_freq_tn["article_id"].map(
            item_to_encoding
        )
        test = pd.concat([clicks_freq_tn, test])
        relevant = p9.relevant(test[test["count"] > 0])

        print(colored("model", "blue"))
        #!model
        # TODO: ajout n_sampled
        model = LightFM(no_components=50, loss="warp", max_sampled=cfg.max_sampled)
        model.fit(train, epochs=cfg.n_epoch, num_threads=20, verbose=True)
        with open("model/lightfm_%s.pkl" % cfg.max_sampled, "wb") as fle:
            pkl.dump(model, fle, protocol=pkl.HIGHEST_PROTOCOL)
        #!prédiction
        test["preds"] = model.predict(
            user_ids=test["user_id"].tolist(),
            item_ids=test["article_id"].tolist(),
            num_threads=20,
        )
        top5 = (
            test.sort_values("user_id")
            .sort_values("preds", ascending=False)
            .groupby("user_id")
            .head(5)
            .groupby("user_id")
            .apply(lambda x: x["article_id"].tolist())
            .reset_index()
        )
        #!params
        params = {
            "model_type": cfg.model_name,
            "n_epoch": cfg.n_epoch,
            "max_sampled": cfg.max_sampled,
        }
        mlflow.log_params(params)

        #!scores
        print(colored("compute score", "red"))
        mapatk = p9.map_at_k(relevant, top5)
        # dictionnaire de scores
        scores = {"mapatk": mapatk}

        #!artefact
        fig, perf = p9.plot_apatk(relevant, top5)
        fig.savefig("plots/apatk_%s.png" % cfg.model_name)
        perf.to_csv("performances/apatk_relevant_%s.csv" % cfg.model_name, index=False)

        #!logging
        mlflow.log_metrics(scores)
        mlflow.log_artifact("model/lightfm_encodings.pkl")
        mlflow.log_artifact("plots/apatk_%s.png" % cfg.model_name)
        mlflow.log_artifact("model/lightfm_%s.pkl" % cfg.max_sampled)


if __name__ == "__main__":
    makerun()
