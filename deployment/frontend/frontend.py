import requests
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import streamlit_authenticator as staut
import yaml
from yaml.loader import SafeLoader

# header
st.title("Article recommender web app")


#!Création de deux tabs
tab1, tab2 = st.tabs(["Get recommendations", "Admin"])

#!Tab utilisateur
with tab1:
    st.header("Top 5 recommendations")
    # Fenêtre d'entrée de l'usr
    user_id = st.text_input("Enter your ID")
    # Bouton de prédiction
    if st.button("Predict top 5 recommendations"):
        if user_id is not None:
            #!requete post http qui va trigger
            data = {"user_id": user_id}
            res = requests.post(
                "https://article-recommender.azurewebsites.net/api/recommend",
                params=data,
            )
            #!recuperation des resultats de la fct
            res = res.json()
            # Si des recos ont été faites, les afficher
            if "top5" in res:
                top5 = res.get("top5")
                c = 1
                for t in top5:
                    st.markdown(str(c) + str(". article ") + str(t))
                    c += 1

            # si aucune reco n'a été renvoyé, afficher l'erreur
            else:
                st.text("Error detail:")
                error = res.get("detail")
                st.text(error)

#!Tab admin
with tab2:
    # Paramètres d'authentification
    with open("config.yaml") as file:
        config = yaml.load(file, Loader=SafeLoader)
    aut = staut.Authenticate(
        config["credentials"], config["cookie"]["name"], config["cookie"]["key"]
    )
    name, status, user_name = aut.login()
    # Si l'authentification est correcte
    if status:
        aut.logout("logout", "main")
        new_usr = st.text_input("Enter a new user ID")
        #!update usr db
        if st.button("Update user database"):
            if new_usr is not None:
                data = {"new_user_id": new_usr}
                #!post on the address generated via azure deployment function
                res = requests.post(
                    "https://article-recommender.azurewebsites.net/api/add_user",
                    params=data,
                )
                #!print message update db
                res = res.json()
                st.text(res.get("message"))
        #!update article db
        new_article = st.text_input("Enter a new article ID")
        if st.button("Update article database"):
            #!update article db
            if new_article is not None:
                data = {"new_article_id": new_article}
                res = requests.post(
                    "https://article-recommender.azurewebsites.net/api/add_article",
                    params=data,
                )
                #!print message update db
                res = res.json()
                st.text(res.get("message"))
    # Si pas d'authentification, la demander
    elif status == False:
        st.error("Username / password is not correct")
    elif status == None:
        st.warning("Plase enter your username and password")
