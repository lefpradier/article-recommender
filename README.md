# [Article RecSys app](https://github.com/lefpradier/article-recommender)
Ce repository contient le code source permettant d'entraîner et de déployer un modèle de RecSys d'articles de presse.
Ce modèle est déployé grâce à une Azure Function (exposée initialement à l'adresse suivante : https://article-recommender.azurewebsites.net/), et utilisé par une application Streamlit (initialement exposée à l'adresse suivante : https://article-recommender-mvp.azurewebsites.net/).
## Spécifications techniques

Le système de recommendation a été entraîné et testé sur le jeu de données [Globo.com](https://www.kaggle.com/datasets/gspmoreira/news-portal-user-interactions-by-globocom). Le modèle choisi pour déploiement est un système de recommendation hybride :
- pour le <i>cold start</i> (nouveaux utilisateurs sans historique) : un système de recommandation par popularité ;
- pour les autres utilisateurs : un algorithme de collaborative filtering BPR (<i>bayesian personalized ranking</i>) à 50 facteurs.
