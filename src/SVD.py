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


@hydra.main(
    version_base=None, config_path="../config", config_name="collaborative_filtering"
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
        clicks_freq = p9.scoring_tfidf(data)
        print(colored("popularity based negative sampling", "red"))
        clicks_freq_tn = p9.add_neg(clicks_freq, 300, popularity)
        print(colored("preprocess and split", "red"))
        dataset, user_item_encodings = preprocess_and_split(clicks_freq_tn)
        pkl.dump(user_item_encodings, open("model/svd_encodings.pkl", "wb"))
        (x_train, y_train), (x_test, y_test) = dataset

        x_test_df = pd.DataFrame(
            {
                "user_id": x_test[:, 0],
                "article_id": x_test[:, 1],
                "score": np.squeeze(y_test),
            }
        )
        # définition des positifs
        relevant = p9.relevant(x_test_df[x_test_df["score"] > 0])

        #!model
        n_users = len(clicks_freq["user_id"].unique())
        n_items = len(clicks_freq["article_id"].unique())
        del dataset
        del user_item_encodings
        del clicks_freq
        del clicks_freq_tn
        gc.collect()
        print(colored("create model and train", "red"))
        model = p9.train_svd(
            x_train, y_train, cfg.batch_size, cfg.n_epoch, n_users, n_items
        )
        model.save("model/svd")
        top5 = p9.predict_top5(model, x_test, cfg.batch_size)

        #!params
        params = {
            "model_type": cfg.model_name,
            "n_epoch": cfg.n_epoch,
            "batch_size": cfg.batch_size,
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
        mlflow.log_artifact("model/svd_encodings.pkl")
        mlflow.log_artifact("model/svd")
        mlflow.log_artifact("plots/apatk_%s.png" % cfg.model_name)
        mlflow.log_artifact("performances/apatk_relevant_%s.csv" % cfg.model_name)


if __name__ == "__main__":
    makerun()
