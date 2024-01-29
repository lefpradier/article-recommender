import os
import numpy as np
import pandas as pd
import random
import warnings
import matplotlib.pyplot as plt
import pickle as pkl

from tfrec.models import SVDpp, SVD
from tfrec.utils import preprocess_and_split
from sklearn.decomposition import PCA
from chunkdot import cosine_similarity_top_k
from termcolor import colored
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import labellines as ll

warnings.filterwarnings("ignore")


def join_data():
    """
    Jointure des fichiers de clics
    """
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
    """
    Conversion d'un dataframe en sparse matrix items x users
    ---
    Paramètres :
    - df: pd.DataFrame
    - users: list
    - items: list
    """
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


#############################
## COLLABORATIVE FILTERING ##
#############################


def scoring(df, thr=0):
    """
    Création de ratings à partir des clics
    ---
    Paramètres :
    - df: pd.DataFrame
    - thr: int, seuil de clics à partir duquel on considère une paire item x user comme positive
    """
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


def add_neg(df, factor, popularity, random_state=42):
    """
    Création de paires item x user négatives
    ---
    Paramètres :
    - df: pd.DataFrame
    - factor: int, nombre de négatifs voulus pour chaque paire positive
    - popularity: pd.DataFrame, tableau de popularité des articles pour pondérer le sampling
    - random_state: int, graine
    """
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
    """
    Entraînement d'un modèle de SVD
    ---
    Paramètres :
    - x_train: np.ndarray
    - y_train: np.ndarray
    - batch_size: int
    - n_epoch: int
    - n_users: int
    - n_items: int
    """
    #!modif de la global mean > ajout de zeros
    global_mean = np.mean(y_train)
    print(colored("create model", "blue"))
    model = SVDpp(n_users, n_items, global_mean, reg_all=1 / 10000)
    # prise en compte des non zeros sur train
    model.implicit_feedback(x_train)
    print(colored("compile model", "blue"))
    model.compile(loss="mean_squared_error", optimizer="adam")
    print(colored("fit model", "blue"))
    model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch)
    return model


def predict_top5(model, x_test, batch_size):
    """
    Prédiction des 5 meilleures prédictions par utilisateur
    ---
    Paramètres :
    - model: modèle tfrec
    - x_test: pd.DataFrame
    - batch_size: int
    """
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


#############
## SCORING ##
#############


def ap_at_k(actual, predicted, k=5):
    """
    Calcul de l'average precision @k
    ---
    Paramètres :
    - actual: list
    - predicted: list
    - k: int
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
    Calcul de la mean average precision @k
    ---
    Paramètres :
    - actual: list de lists
    - predicted: list de lists
    - k: int
    """
    common = actual.merge(predicted, on="user_id")
    actual = common["article_id"].values.tolist()
    predicted = common[0].values.tolist()
    return np.mean([ap_at_k(a, p, k) for a, p in zip(actual, predicted)])


def plot_apatk(relevant, top5):
    """
    Création d'une figure représentant l'évolution de l'AP@k en fonction
    du nombre d'articles lus par utilisateur
    ---
    Paramètres :
    - relevant: pd.DataFrame
    - top5: list de lists
    """
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


def plot_hyperparam(df):
    """
    Création d'une figure de résultats de l'hyperparamétrisation
    ---
    Paramètres :
    - df: pd.DataFrame
    """
    plt.style.use("custom_dark")
    fig, ax = plt.subplots(1, 1)
    for n in df["n_factor"].unique():
        ax.plot(
            df.loc[df["n_factor"] == n, "thr_pop"],
            df.loc[df["n_factor"] == n, "mapatk"],
            label=str(n),
        )
    fig.legend(
        [int(i) for i in df["n_factor"].unique()],
        title="Number of BPR factors",
        borderaxespad=3,
    )
    ax.set_xlabel("Number of reads above which to use BPR")
    ax.set_ylabel("MAP@5")


###################
## CONTENT BASED ##
###################


def get_similarity(top_k=5, pca_n_comp=25, red=True):
    """
    Création d'une matrice de similarité par cosinus, éventuellement après ACP
    sur la matrice d'embedding
    ---
    Paramètres :
    - top_k: int
    - pca_n_comp: int
    - red: bool, applique l'ACP uniquement si True
    """
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
    """
    Crée une figure montrant la similarité moyenne entre articles lus par utilisateur
    ---
    Paramètres :
    - df: pd.DataFrame, tableau d'articles lus par utilisateur
    """

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


################
## POPULARITÉ ##
################


def get_popularity(data):
    """
    Calcule la popularité de chaque article comme le
    nombre d'utilisateurs ayant déjà lu l'article
    ---
    Paramètres :
    - data: pd.DataFrame
    """
    popdf = data.groupby("click_article_id").size().reset_index()
    popdf.rename(
        columns={0: "n_clicks", "click_article_id": "article_id"}, inplace=True
    )
    popdf.sort_values(by="n_clicks", ascending=False, inplace=True)
    return popdf


def relevant(df):
    """
    Récupère la liste d'articles lus par utilisateur
    ---
    Paramètres :
    - df: pd.DataFrame
    """
    relevant = (
        df.sort_values("user_id")
        .groupby("user_id")["article_id"]
        .apply(list)
        .reset_index()
    )  # liste par usr contenues dans une liste globale
    return relevant
