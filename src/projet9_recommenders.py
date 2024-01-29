import numpy as np
import pandas as pd
import warnings
from itertools import product
from scipy.sparse import csr_matrix
from tqdm import tqdm
from alive_progress import alive_bar
from termcolor import colored

warnings.filterwarnings("ignore")


class cf_recommender:
    """
    Algorithme de recommandation par collaborative filtering
    """

    def __init__(self, model, idx_usr, idx_it_ref, idx_it_mod):
        """
        Paramètres :
        - model: modèle de collaborative filtering
        - idx_usr: dict, référence des identifiants user dans le modèle
        - idx_it_ref: dict, référence des identifiants item réels
        - idx_it_mod: dict, références des identifiants item dans le modèle
        """
        self.model = model
        self.idx_usr = idx_usr
        self.idx_it_ref = idx_it_ref
        self.idx_it_mod = idx_it_mod

    def input(self, user, items):
        """
        Paramètres :
        - user: identifiant utilisateur pour lequel on fait des prédictions
        - items: identifiants des items déjà lus par l'utilisateur
        """
        # conversion des identifiants
        user_model = self.idx_usr[user]
        items_model = [self.idx_it_mod[item] for item in items]
        # liste des items dont on veut tester le score de recommandation
        items_to_test = [
            item for item in range(max(self.idx_it_ref) + 1) if item not in items_model
        ]
        # conversion en dataframe
        it_usr_model = pd.DataFrame(
            {"user": [user_model] * len(items_to_test), "item": items_to_test}
        )
        return it_usr_model

    def predict(self, it_usr_model):
        """
        Paramètres :
        - it_usr_model : pd.DataFrame, tableau user x items sur lequel prédire des scores
        """
        # prédiction
        it_usr_array = it_usr_model.to_numpy()
        it_usr_model["preds"] = self.model.predict(it_usr_array)
        # ranking
        top5 = it_usr_model.sort_values("preds", ascending=False)["item"].head(5)
        return top5

    def recommend(self, user, items):
        """
        Paramètres :
        - user : identifiant utilisateur pour lequel on fait des recommandations
        - items : identifiants des items déjà lus
        """
        # création des inputs du modèle
        it_usr_model = self.input(user, items)
        # prédiction des 5 meilleurs items
        top5 = self.predict(it_usr_model)
        # reconversion des items recommandés vers les identifiants de référence
        top5 = [self.idx_it_ref[item] for item in top5]
        return top5


class pop_recommender:
    """
    Algorithme de recommandation par popularité
    """

    def __init__(self, pop):
        """
        Paramètres :
        - pop : pd.DataFrame, popularité des articles
        """
        self.popularity = pop

    def recommend(self, lu, k=5):
        """
        Paramètres :
        - lu: liste des articles déjà lus par chaque utilisateur
        - k: int, nombre d'articles à recommander
        """
        # réduction de la liste pour diminuer la complexité de calcul
        maxread = lu.groupby("user_id").size().max()
        sub_pop = self.popularity.head(maxread + k)
        # prédiction des top k articles par utilisateur
        non_lu = (
            lu.groupby("user_id")
            .apply(
                lambda x: sub_pop[~sub_pop["article_id"].isin(x["article_id"])].head(k)
            )
            .reset_index()
        )
        top_k = (
            non_lu.sort_values("user_id")
            .groupby("user_id")
            .apply(lambda x: x["article_id"].tolist())
            .reset_index()
        )
        return top_k


class cb_recommender:
    """
    Algorithme de recommandation basé sur le contenu des articles
    """

    def __init__(self, sim):
        """
        Paramètres :
        - sim : matrice de similarité entre articles
        """
        self.sim = sim

    def recommend(self, lu):
        """
        Paramètres :
        - lu: liste des articles déjà lus par chaque utilisateur
        """
        sim_df_lu = self.sim.merge(
            lu, left_on="ref_item", right_on="article_id", how="right"
        )
        #!score de ranking
        sim_df_lu["score_rank"] = sim_df_lu["similarity"] * sim_df_lu["freq"]
        #!colonne comp item sup ceux de ref_item
        sim_df_lu = (
            sim_df_lu.groupby("user_id")
            .apply(
                lambda x: x[~x["compared_item"].isin(x["ref_item"])].drop(
                    columns="user_id"
                )
            )
            .reset_index()
        )
        #!somme des scores pour des items présent sur plusieurs lignes de reco
        top5 = (
            sim_df_lu.groupby(["user_id", "compared_item"], observed=True)
            .agg({"score_rank": "sum"})
            .reset_index()
            .sort_values(by="score_rank", ascending=False)
            .groupby("user_id")
            .head(5)
            .groupby("user_id")
            .apply(lambda x: x["compared_item"].tolist())
            .reset_index()
        )
        return top5


class hybrid_recommender_bpr:
    """
    Algorithme de recommandation hybride dont l'algorithme de
    collaborative filtering est un Bayesian Personalized ranking
    """

    def __init__(
        self, model_cf, model_pop, reads, thr_pop=0, thr_cb=0, model_content=None
    ):
        """
        Paramètres :
        - model_cf: modèle BPR
        - model_pop: pd.DataFrame, popularité des articles
        - model_content: matrice de similarité entre articles
        - reads: dict, users et items pris en compte dans le BPR
        - thr_pop: int, seuil de nombre d'articles lus au dessus duquel on n'applique pas l'algorithme de popularité
        - thr_cb: int, seuil de nombre d'articles lus au dessus duquel on n'applique pas l'algorithme content-based
        """
        self.model_cf = model_cf
        self.model_pop = model_pop
        self.model_content = model_content
        self.thr_cb = thr_cb
        self.thr_pop = thr_pop
        self.reads = reads

    def __recommend_pop(self, lu, k=5):
        """
        Recommandation par popularité
        ---
        Paramètres :
        - lu: liste d'articles déjà lus par utilisateur
        - k: int, nombre de recommandations à effectuer
        """
        maxread = lu.groupby("user_id").size().max()
        sub_pop = self.model_pop.head(maxread + k)
        non_lu = (
            lu.groupby("user_id")
            .apply(
                lambda x: sub_pop[~sub_pop["article_id"].isin(x["article_id"])].head(k)
            )
            .reset_index()
        )
        top_k = (
            non_lu.sort_values("user_id")
            .groupby("user_id")
            .apply(lambda x: x["article_id"].tolist())
            .reset_index()
        )
        return top_k

    def __input(self, user_items):
        """
        Mise en forme des données pour l'algorithme BPR
        ---
        Paramètres :
        - user_items: pd.DataFrame, tableau users x items lus
        """
        row = pd.Categorical(user_items["user_id"], categories=self.reads["usrs"]).codes
        col = pd.Categorical(
            user_items["article_id"], categories=self.reads["items"]
        ).codes
        spm = csr_matrix(
            ([1 for i in range(len(user_items.index))], (row, col)),
            shape=(len(self.reads["usrs"]), len(self.reads["items"])),
        )
        return spm

    def __predict(self, spm):
        """
        Prédiction des recommandations par l'algorithme BPR
        ---
        Paramètres :
        - spm: sparse matrix items x users
        """
        usrs = spm.shape[0]
        usrs_idx = np.arange(usrs)
        usrs_idx = usrs_idx[np.ediff1d(spm.indptr) > 0]
        # prédiction
        preds, _ = self.model_cf.recommend(
            userid=usrs_idx, user_items=spm[usrs_idx], N=5
        )
        top5 = pd.DataFrame({"user_id": usrs_idx})
        top5[0] = pd.Series(map(lambda x: [x], preds)).apply(lambda x: x[0])
        return top5

    def __recommend_cf_step(self, user_items):
        """
        Prédiction avec l'algorithme BPR sur un batch d'utilisateurs
        ---
        Paramètres :
        - user_items: pd.DataFrame, tableau items x users
        """
        it_usr_model = self.__input(user_items)
        top5 = self.__predict(it_usr_model)
        return top5

    def __recommend_cf(self, lu, k=5, step=400000):
        """
        Recommandation par l'algorithme BPR et organisation en batchs
        ---
        Paramètres :
        - lu: liste des articles déjà lus par les utilisateurs
        - k: int, nombre de recommandations à faire
        - step: int, taille des batchs
        """
        top_k = []
        all_users = lu["user_id"].unique().tolist()
        with tqdm(len(all_users)) as bar:
            for i in range(0, len(all_users), step):
                usrs = all_users[i : (i + step)]
                top5 = self.__recommend_cf_step(lu[lu["user_id"].isin(usrs)])
                top_k.append(top5)
                bar.update(step)

        top_k = pd.concat(top_k)
        return top_k

    def __recommend_cb(self, lu, k=5):
        """
        Recommandation par l'algorithme basé sur la similarité entre articles
        ---
        Paramètres :
        - lu: liste des articles déjà lus par les utilisateurs
        - k: nombre de recommandations à effectuer
        """
        sim_df_lu = self.model_cb.merge(
            lu, left_on="ref_item", right_on="article_id", how="right"
        )
        #!score de ranking
        sim_df_lu["score_rank"] = sim_df_lu["similarity"] * sim_df_lu["count"]
        #!colonne comp item sup ceux de ref_item
        sim_df_lu = (
            sim_df_lu.groupby("user_id")
            .apply(
                lambda x: x[~x["compared_item"].isin(x["ref_item"])].drop(
                    columns="user_id"
                )
            )
            .reset_index()
        )
        #!somme des scores pour des items présent sur plusieurs lignes de reco
        top_k = (
            sim_df_lu.groupby(["user_id", "compared_item"], observed=True)
            .agg({"score_rank": "sum"})
            .reset_index()
            .sort_values(by="score_rank", ascending=False)
            .groupby("user_id")
            .head(5)
            .groupby("user_id")
            .apply(lambda x: x["compared_item"].tolist())
            .reset_index()
        )
        return top_k

    def recommend(self, lu):
        """
        Découpage des utilisateurs en fonction du nombre d'articles lus
        et application des différents algorithmes de recommandation
        ---
        Paramètres :
        - lu: liste des articles déjà lus par les différents utilisateurs
        """
        # comptage du nombre d'articles lus par chaque utilisateur
        n_user = lu.groupby("user_id").size().reset_index().rename(columns={0: "count"})
        # découpage en plusieurs subsets en fonctions des seuils thr_pop et thr_cb
        users_pop = n_user.loc[n_user["count"] < self.thr_pop, :]
        users_cb = n_user.loc[n_user["count"] < self.thr_cb, :]
        users_cf = n_user.loc[
            (n_user["count"] >= self.thr_cb) & (n_user["count"] >= self.thr_pop), :
        ]
        top_k = []
        # application de l'algorithme de popularité
        if users_pop.shape[0] > 0:
            print(colored("apply popularity recommender", "yellow"))
            topk_pop = self.__recommend_pop(
                lu[lu["user_id"].isin(users_pop["user_id"])]
            )
            topk_pop["method"] = "popularity"
            top_k.append(topk_pop)
        # application de l'algorithme content-based
        if users_cb.shape[0] > 0 and self.model_content is not None:
            print(colored("apply content-based recommender", "yellow"))
            topk_cb = self.__recommend_cb(lu[lu["user_id"].isin(users_cb["user_id"])])
            topk_cb["method"] = "content_based"
            top_k.append(topk_cb)
        # application de l'algorithme BPR
        if users_cf.shape[0] > 0:
            print(colored("apply collaborative filtering recommender", "yellow"))
            topk_cf = self.__recommend_cf(lu[lu["user_id"].isin(users_cf["user_id"])])
            topk_cf["method"] = "collaborative_filtering"
            top_k.append(topk_cf)
        top_k = pd.concat(top_k)
        return top_k
