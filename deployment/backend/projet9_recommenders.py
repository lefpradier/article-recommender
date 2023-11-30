import numpy as np
import pandas as pd
import warnings
from itertools import product
from scipy.sparse import csr_matrix
from tqdm import tqdm
from termcolor import colored

warnings.filterwarnings("ignore")


class cf_recommender:
    # appliquer et deployer le modèle de collaborative filtering
    def __init__(self, model, idx_usr, idx_it_ref, idx_it_mod):
        self.model = model
        self.idx_usr = idx_usr
        self.idx_it_ref = idx_it_ref
        self.idx_it_mod = idx_it_mod

    def input(self, user, items):
        user_model = self.idx_usr[user]
        items_model = [self.idx_it_mod[item] for item in items]
        #!2.liste des trucs qu'on veut tester (tous it sauf ceux deja lu)
        items_to_test = [
            item for item in range(max(self.idx_it_ref) + 1) if item not in items_model
        ]
        #!3.convert idx_it et idx_usr vers model
        it_usr_model = pd.DataFrame(
            {"user": [user_model] * len(items_to_test), "item": items_to_test}
        )
        return it_usr_model

    def predict(self, it_usr_model):
        #!4.prediction sur la db
        it_usr_array = it_usr_model.to_numpy()
        it_usr_model["preds"] = self.model.predict(it_usr_array)
        #!6.ranking
        top5 = it_usr_model.sort_values("preds", ascending=False)["item"].head(5)
        return top5

    def recommend(self, user, items):
        it_usr_model = self.input(user, items)
        top5 = self.predict(it_usr_model)
        #!5.reconversion des idx_it vers ceux de la db init
        top5 = [self.idx_it_ref[item] for item in top5]
        return top5


class pop_recommender:
    def __init__(self, pop):
        self.popularity = pop

    def recommend(self, lu, k=5):
        maxread = lu.groupby("user_id").size().max()
        sub_pop = self.popularity.head(maxread + k)
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


# class qui a partir de l'embedding renvoie des predictions
class cb_recommender:
    def __init__(self, sim):
        self.sim = sim

    def recommend(self, lu):
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


class hybrid_recommender_lightfm:
    def __init__(
        self,
        model_cf,
        model_pop,
        model_content,
        idx_usr,
        idx_it_ref,
        idx_it_mod,
        thr_pop,
        thr_cb,
    ):
        self.model_cf = model_cf
        self.model_pop = model_pop
        self.model_content = model_content
        self.idx_usr = idx_usr
        self.idx_it_ref = idx_it_ref
        self.idx_it_mod = idx_it_mod
        self.thr_cb = thr_cb
        self.thr_pop = thr_pop

    def __recommend_pop(self, lu, k=5):
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
        user_items["user_id"] = user_items["user_id"].map(self.idx_usr)
        user_items["article_id"] = user_items["article_id"].map(self.idx_it_mod)
        #!2.liste des trucs qu'on veut tester (tous it sauf ceux deja lu)
        items_to_test = pd.DataFrame(
            list(
                product(
                    user_items["user_id"].unique().tolist(),
                    list(self.idx_it_ref.keys()),
                )
            ),
            columns=["user_id", "article_id"],
        )
        items_to_test = items_to_test.merge(
            user_items, on=["user_id", "article_id"], how="left", indicator=True
        )
        items_to_test = items_to_test.loc[
            items_to_test["_merge"] == "left_only", ["user_id", "article_id"]
        ]
        return items_to_test

    def __predict(self, it_usr_model):
        #!4.prediction sur la db
        # it_usr_array = it_usr_model.to_numpy()
        it_usr_model["preds"] = self.model_cf.predict(
            user_ids=it_usr_model["user_id"].tolist(),
            item_ids=it_usr_model["article_id"].tolist(),
            num_threads=20
            # it_usr_array, batch_size=max(self.idx_usr)
        )
        #!6.ranking
        top5 = (
            it_usr_model.sort_values("preds", ascending=False)
            .groupby("user_id")
            .head(5)
        )
        top5["article_id"] = top5["article_id"].map(self.idx_it_ref)
        top5 = (
            top5.groupby("user_id")
            .apply(lambda x: x["article_id"].tolist())
            .reset_index()
        )
        return top5

    def __recommend_cf_step(self, user_items):
        it_usr_model = self.__input(user_items)
        top5 = self.__predict(it_usr_model)
        #!5.reconversion des idx_it vers ceux de la db init
        return top5

    def __recommend_cf(self, lu, k=5, step=1000):
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
        sim_df_lu = self.model_cb.merge(
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
        n_user = lu.groupby("user_id").size().reset_index().rename(columns={0: "count"})
        users_pop = n_user.loc[n_user["count"] < self.thr_pop, :]
        users_cb = n_user.loc[n_user["count"] < self.thr_cb, :]
        users_cf = n_user.loc[
            (n_user["count"] >= self.thr_cb) & (n_user["count"] >= self.thr_pop), :
        ]
        top_k = []
        if users_pop.shape[0] > 0:
            topk_pop = self.__recommend_pop(
                lu[lu["user_id"].isin(users_pop["user_id"])]
            )
            topk_pop["method"] = "popularity"
            top_k.append(topk_pop)
        if users_cb.shape[0] > 0:
            topk_cb = self.__recommend_cb(lu[lu["user_id"].isin(users_cb["user_id"])])
            topk_cb["method"] = "content_based"
            top_k.append(topk_cb)
        if users_cf.shape[0] > 0:
            topk_cf = self.__recommend_cf(lu[lu["user_id"].isin(users_cf["user_id"])])
            topk_cf["method"] = "collaborative_filtering"
            top_k.append(topk_cf)
        top_k = pd.concat(top_k)
        return top_k


class hybrid_recommender_bpr:
    def __init__(
        self, model_cf, model_pop, reads, thr_pop=0, thr_cb=0, model_content=None
    ):
        self.model_cf = model_cf
        self.model_pop = model_pop
        self.model_content = model_content
        self.thr_cb = thr_cb
        self.thr_pop = thr_pop
        self.reads = reads

    def __recommend_pop(self, lu, k=5):
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
        # user_dict = {all_users[i]: i for i in range(len(all_users))}
        # dict_user = {i: all_users[i] for i in range(len(all_users))}
        # user_items["user_id"] = user_items["user_id"].map(user_dict)
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
        usrs = spm.shape[0]
        usrs_idx = np.arange(usrs)
        usrs_idx = usrs_idx[np.ediff1d(spm.indptr) > 0]
        #!4.prediction sur la db
        preds, _ = self.model_cf.recommend(
            userid=usrs_idx, user_items=spm[usrs_idx], N=5
        )
        top5 = pd.DataFrame({"user_id": usrs_idx})
        top5[0] = pd.Series(map(lambda x: [x], preds)).apply(lambda x: x[0])
        return top5

    def __recommend_cf_step(self, user_items):
        it_usr_model = self.__input(user_items)
        top5 = self.__predict(it_usr_model)
        #!5.reconversion des idx_it vers ceux de la db init
        return top5

    def __recommend_cf(self, lu, k=5, step=400000):
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
        n_user = lu.groupby("user_id").size().reset_index().rename(columns={0: "count"})
        users_pop = n_user.loc[n_user["count"] < self.thr_pop, :]
        users_cb = n_user.loc[n_user["count"] < self.thr_cb, :]
        users_cf = n_user.loc[
            (n_user["count"] >= self.thr_cb) & (n_user["count"] >= self.thr_pop), :
        ]
        top_k = []
        if users_pop.shape[0] > 0:
            print(colored("apply popularity recommender", "yellow"))
            topk_pop = self.__recommend_pop(
                lu[lu["user_id"].isin(users_pop["user_id"])]
            )
            topk_pop["method"] = "popularity"
            top_k.append(topk_pop)
        if users_cb.shape[0] > 0 and self.model_cb is not None:
            print(colored("apply content-based recommender", "yellow"))
            topk_cb = self.__recommend_cb(lu[lu["user_id"].isin(users_cb["user_id"])])
            topk_cb["method"] = "content_based"
            top_k.append(topk_cb)
        if users_cf.shape[0] > 0:
            print(colored("apply collaborative filtering recommender", "yellow"))
            topk_cf = self.__recommend_cf(lu[lu["user_id"].isin(users_cf["user_id"])])
            topk_cf["method"] = "collaborative_filtering"
            top_k.append(topk_cf)
        top_k = pd.concat(top_k)
        return top_k
