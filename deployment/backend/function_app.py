import azure.functions as func
import logging
from io import BytesIO

# from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import pickle as pkl
import pandas as pd
import json
import projet9_recommenders as reco
from termcolor import colored

# creation API serverless
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

connection_string = "AzureWebJobsStorage"


#!ROUTE RECO
#!déclenchement de la prediction
@app.function_name(name="recommend")
@app.route(route="recommend", auth_level=func.AuthLevel.ANONYMOUS)
#!ajout decorateur pour connexion avec blob storage
@app.blob_input(
    arg_name="modelpop",
    path="recommender/popularity.csv",
    connection=connection_string,
    data_type="binary",
)
@app.blob_input(
    arg_name="userdb",
    path="recommender/user_db.pkl",
    connection=connection_string,
    data_type="binary",
)
@app.blob_input(
    arg_name="bprreads",
    path="recommender/bpr_reads.pkl",
    connection=connection_string,
    data_type="binary",
)
@app.blob_input(
    arg_name="modelbpr",
    path="recommender/bpr.pkl",
    connection=connection_string,
    data_type="binary",
)
def predict(
    req: func.HttpRequest,
    modelpop: func.InputStream,
    modelbpr: func.InputStream,
    bprreads: func.InputStream,
    userdb: func.InputStream,
) -> str:
    logging.info("Python HTTP trigger function processed a request.")
    usr = req.params.get("user_id")
    if not usr:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            usr = req_body.get("user_id")
    if usr:
        ####!INSERT MODEL AND PREDICTION ACTIONS #####
        user_db = pkl.loads(userdb.read())
        if int(usr) in user_db.keys():
            bpr = pkl.loads(modelbpr.read())
            bpr_reads = pkl.loads(bprreads.read())
            pop = pd.read_csv(BytesIO(modelpop.read()))
            recommender = reco.hybrid_recommender_bpr(
                model_cf=bpr, model_pop=pop, reads=bpr_reads
            )
            lu = pd.DataFrame({"user_id": [], "article_id": []})
            lu["article_id"] = user_db[int(usr)]
            lu["user_id"] = int(usr)
            # revoie json avec top5
            top5 = recommender.recommend(lu)
            top5 = top5[0].values[0].tolist()
            res = {"user_id": usr, "top5": top5}
        else:
            res = {
                "detail": "user unknown in the current database, please add it as admin"
            }
    else:
        res = {"detail": "please provide a user id"}
    return json.dumps(res)


#!ROUTE ADMIN USER
#!déclenchement de la prediction
@app.function_name(name="add_user")
@app.route(route="add_user", auth_level=func.AuthLevel.ANONYMOUS)
#!blob input and output
@app.blob_input(
    arg_name="userdb",
    path="recommender/user_db.pkl",
    connection=connection_string,
    data_type="binary",
)
@app.blob_output(
    arg_name="newuserdb",
    path="recommender/user_db.pkl",
    connection=connection_string,
    data_type="binary",
)
def add_user(
    req: func.HttpRequest,
    userdb: func.InputStream,
    newuserdb: func.Out[func.InputStream],
) -> str:
    add_usr = req.params.get("new_user_id")
    if not add_usr:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            add_usr = req_body.get("new_user_id")
    if add_usr:
        #!load db
        user_db = pkl.loads(userdb.read())
        #! si new user
        if int(add_usr) not in user_db.keys():
            # update new usr
            user_db[int(add_usr)] = []
            # upload la db updated vers azure
            user_db_updated = pkl.dumps(user_db)
            newuserdb.set(user_db_updated)
            return json.dumps({"message": "user db updated"})
        else:
            return json.dumps({"message": "user already exists in db"})
    else:
        return json.dumps({"message": "please provide a user id"})


#!ROUTE ADMIN ARTICLE
#!déclenchement de la prediction
@app.function_name(name="add_article")
@app.route(route="add_article", auth_level=func.AuthLevel.ANONYMOUS)
#!blob input and output
@app.blob_input(
    arg_name="articledb",
    path="recommender/article_db.pkl",
    connection=connection_string,
    data_type="binary",
)
@app.blob_output(
    arg_name="newarticledb",
    path="recommender/article_db.pkl",
    connection=connection_string,
    data_type="binary",
)
def add_article(
    req: func.HttpRequest,
    articledb: func.InputStream,
    newarticledb: func.Out[func.InputStream],
) -> str:
    add_item = req.params.get("new_article_id")
    if not add_item:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            add_item = req_body.get("new_article_id")
    if add_item:
        #!load db
        article_db = pkl.loads(articledb.read())
        #! si new user
        if int(add_item) not in article_db:
            # update new usr
            article_db.append(int(add_item))
            # upload la db updated vers azure
            article_db_updated = pkl.dumps(article_db)
            newarticledb.set(article_db_updated)
            return json.dumps({"message": "article db updated"})
        else:
            return json.dumps({"message": "article already exists in db"})
    else:
        return json.dumps({"message": "please enter a new article id"})
