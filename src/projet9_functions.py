import os
import numpy as np
import pandas as pd
import random
import warnings
import matplotlib.pyplot as plt
import pickle as pkl

# from tfrec.models import SVDpp, SVD
# from tfrec.utils import preprocess_and_split
from sklearn.decomposition import PCA
from chunkdot import cosine_similarity_top_k
from termcolor import colored
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")


# Jointure des fichiers de click
def join_data():
    clicks_path = [
        f
        for f in os.listdir("data/clicks")
        if os.path.isfile(os.path.join("data/clicks", f))
    ]

    clicks_path.sort()
    list = []

    for filename in clicks_path:
        df = pd.read_csv(
            os.path.join("data/clicks", filename), index_col=None, header=0
        )
        list.append(df)

    clicks = pd.concat(list, axis=0, ignore_index=True)
    clicks.to_csv("data/clicks.csv")
    print("jointure des fichiers effectuée")


def convert_to_spm(df, users=None, items=None):
    # creation de la sparse matrix
    if users is None:
        users = list(df["user_id"].unique())
    if items is None:
        items = list(df["article_id"].unique())
    users.sort()
    items.sort()
    row = pd.Categorical(df["user_id"], categories=users).codes
    col = pd.Categorical(df["article_id"], categories=items).codes
    spm = csr_matrix((df["count"].tolist(), (row, col)), shape=(len(users), len(items)))
    return spm


##COLLABORATIVE FILTERING


# creation des ratings
def scoring(df, thr=0):
    # count des clicks/items/usr
    clicks_itusr = df.groupby(["user_id", "click_article_id"]).size().reset_index()
    clicks_itusr.rename(
        columns={0: "count", "click_article_id": "article_id"}, inplace=True
    )
    clicks_itusr = clicks_itusr[
        clicks_itusr.groupby("user_id")["count"].transform("sum") > thr
    ]
    # count des clicks/usr
    clicks_itusr["count"] /= clicks_itusr.groupby("user_id")["count"].transform("sum")
    return clicks_itusr  # freq item/usr


#!implzementation scoring tfidf
def scoring_tfidf(df):
    # compute item-frequencies (tf)
    tf = df.groupby(["user_id", "click_article_id"]).size().reset_index()
    tf.rename(columns={0: "count", "click_article_id": "article_id"}, inplace=True)
    tf["count"] /= tf.groupby("user_id")["count"].transform("sum")
    # compute inverse user-frequence (idf)
    idf = (
        df.groupby("click_article_id")
        .agg({"user_id": "nunique"})
        .reset_index()
        .rename(columns={"click_article_id": "article_id", "user_id": "nuser"})
    )
    idf["idf"] = np.log(df["user_id"].nunique() / idf["nuser"])
    tf_idf = tf.merge(idf, on="article_id")
    # tfidf/usr/item
    tf_idf["score"] = tf_idf["count"] * tf_idf["idf"]
    # subset columns
    tf_idf = tf_idf[["user_id", "article_id", "score"]]
    return tf_idf  # freq item/usr


# ajout de TN pour les df


def add_neg(df, factor, popularity, random_state=42):
    random.seed(random_state)
    #!genere des comb rd item/usr
    #!introduction popularity based negative sampling pour le SVD
    print(colored("random combinations", "blue"))
    tn = np.array(
        [
            random.choices(
                df["user_id"].unique(), k=factor * len(df["user_id"].unique())
            ),
            random.choices(
                popularity["article_id"].tolist(),
                k=factor * len(df["user_id"].unique()),
                weights=popularity["n_clicks"].tolist(),
            ),
        ]
    )
    tn = np.transpose(tn)
    # filtrage des combinaisons par rapport à ce qui est déjà connu
    tn = pd.DataFrame(tn, columns=["user_id", "article_id"])
    print(colored("merge", "blue"))
    tn = tn.merge(
        df[["user_id", "article_id"]],
        on=["user_id", "article_id"],
        how="left",
        indicator=True,
    )
    tn = tn.loc[tn["_merge"] == "left_only", ["user_id", "article_id"]]
    tn["count"] = 0
    df_tn = pd.concat([df, tn])
    return df_tn


def train_svd(x_train, y_train, batch_size, n_epoch, n_users, n_items):
    #!modif de la global mean > ajout de zeros
    global_mean = np.mean(y_train)
    print(colored("create model", "blue"))
    # model = SVDpp(n_users, n_items, global_mean, reg_all=1 / 10000)
    model = SVD(n_users, n_items, global_mean, reg_all=1 / 10000)
    # prise en compte des non zeros sur train
    # print(colored("implicit feedback preheat", "blue"))
    # model.implicit_feedback(x_train)
    print(colored("compile model", "blue"))
    model.compile(loss="mean_squared_error", optimizer="adam")
    print(colored("fit model", "blue"))
    model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch)
    return model


# predcition des top5
def predict_top5(model, x_test, batch_size):
    x_test_df = pd.DataFrame(x_test, columns=["user_id", "article_id"])
    x_test_df["preds"] = model.predict(x_test, batch_size=batch_size)
    top5 = (
        x_test_df.sort_values("user_id")
        .sort_values("preds", ascending=False)
        .groupby("user_id")
        .head(5)
        .groupby("user_id")
        .apply(lambda x: x["article_id"].tolist())
        .reset_index()
    )
    return top5


# Scoring des predictions
def ap_at_k(actual, predicted, k=5):
    """
    Computes the average precision at k.

    This function computes the average prescision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def map_at_k(actual, predicted, k=5):
    """
    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists
    of lists of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists

    """
    common = actual.merge(predicted, on="user_id")
    actual = common["article_id"].values.tolist()
    predicted = common[0].values.tolist()
    return np.mean([ap_at_k(a, p, k) for a, p in zip(actual, predicted)])


# etude lien entre relevance ett performances
def plot_apatk(relevant, top5):
    common = relevant.merge(top5, on="user_id")
    relevant = common["article_id"].values.tolist()
    top5 = common[0].values.tolist()
    ap = [ap_at_k(rel, top, k=5) for rel, top in zip(relevant, top5)]
    lens = [len(rel) for rel in relevant]
    perf_by_len = pd.DataFrame({"n_relevant": lens, "ap_at_k": ap})

    plt.style.use("custom_dark")
    fig, ax = plt.subplots(1, 1)
    ax.scatter(lens, ap)
    ax.set_xlabel("Nombre d'article consultés")
    ax.set_ylabel("ap@5")
    return fig, perf_by_len


##CONTENT BASED


# ACP sur l'embedding et calcul des similarités de contenu via cosinus
def get_similarity(top_k=5, pca_n_comp=25, red=True):
    emb = pkl.load(open("data/articles_embeddings.pickle", "rb"))
    if red:
        pca = PCA(n_components=pca_n_comp)
        emb = pca.fit_transform(emb)
    sim = cosine_similarity_top_k(emb, top_k=top_k, max_memory=20e9, show_progress=True)
    sim_coo = sim.tocoo(copy=False)
    sim_df = pd.DataFrame(
        {
            "ref_item": sim_coo.row,
            "compared_item": sim_coo.col,
            "similarity": sim_coo.data,
        }
    )
    return sim_df


def get_average_similarity(df):
    def upper_tri_masking(x):
        m = x.shape[0]
        r = np.arange(m)
        mask = r[:, None] < r
        return x[mask]

    df = scoring(df)
    emb = pkl.load(open("data/articles_embeddings.pickle", "rb"))
    avg_cosine = []
    users = df["user_id"].unique().tolist()
    for usr in users:
        cosine = upper_tri_masking(
            cosine_similarity(
                emb[df.loc[df["user_id"] == usr, "article_id"].unique().tolist(), :]
            )
        )
        avg_cosine.append(np.mean(cosine))
    avgcos = pd.DataFrame(
        {
            "id": users,
            "cos": avg_cosine,
            "n": df.groupby("user_id")["article_id"].nunique().tolist(),
        }
    )
    plt.style.use("custom_dark")
    plt.hist2d(avgcos["n"], avgcos["cos"], bins=(1000, 100))
    plt.ylabel("Avg cosine similarity per user")
    plt.xlabel("Number of articles per user")
    plt.xscale("log")


##POPULARITE


def get_popularity(data):
    popdf = data.groupby("click_article_id").size().reset_index()
    popdf.rename(
        columns={0: "n_clicks", "click_article_id": "article_id"}, inplace=True
    )
    popdf.sort_values(by="n_clicks", ascending=False, inplace=True)
    return popdf


def relevant(df):
    relevant = (
        df.sort_values("user_id")
        .groupby("user_id")["article_id"]
        .apply(list)
        .reset_index()
    )  # liste par usr contenues dans une liste globale
    return relevant
