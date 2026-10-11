"""Interface Streamlit (en francais) : detection hybride d'URL de phishing.
Usage : streamlit run app.py
Ne visite jamais les URL saisies : analyse lexicale locale uniquement.
"""
import json
import os
import re
from urllib.parse import urlparse

import joblib
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE, "models")
DATA_PATH = os.path.join(BASE, "data", "dataset.csv")


def extract_lexical_features(url: str) -> dict:
    s = str(url).strip()
    low = s.lower()
    try:
        host = urlparse(s if "://" in s else "http://" + s).netloc.lower()
    except Exception:
        host = ""
    ip_pattern = r"^(\d{1,3}\.){3}\d{1,3}$"
    host_clean = host.split(":")[0]
    return {
        "url_length": len(s),
        "host_length": len(host),
        "nb_dots": s.count("."),
        "nb_hyphens": s.count("-"),
        "nb_slashes": s.count("/"),
        "nb_digits": sum(c.isdigit() for c in s),
        "nb_special": sum(s.count(c) for c in ["@", "?", "=", "%", "&", "_", "~"]),
        "has_https": 1 if low.startswith("https") else 0,
        "has_ip": 1 if re.match(ip_pattern, host_clean) else 0,
        "nb_subdomains": max(0, host.count(".") - 1) if "." in host else 0,
    }


@st.cache_resource
def load_models():
    # Securite : joblib charge uniquement nos propres fichiers models/*.pkl
    # generes localement par train.py (source de confiance), jamais un URL distant.
    paths = {k: os.path.join(MODELS_DIR, f)
             for k, f in {"scaler": "scaler.pkl", "knn": "knn.pkl",
                          "kmeans": "kmeans.pkl", "hybrid": "knn_hybrid.pkl",
                          "meta": "meta.json"}.items()}
    if not all(os.path.exists(p) for p in [paths["scaler"], paths["knn"],
                                           paths["kmeans"], paths["hybrid"], paths["meta"]]):
        return None
    return {
        "scaler": joblib.load(paths["scaler"]),
        "knn": joblib.load(paths["knn"]),
        "kmeans": joblib.load(paths["kmeans"]),
        "hybrid": joblib.load(paths["hybrid"]),
        "meta": json.load(open(paths["meta"], encoding="utf-8")),
    }


def build_vector(url: str, meta) -> pd.DataFrame:
    """Construit le vecteur de features pour une URL saisie.
    Note : si le dataset reel contient des colonnes numeriques non derivables
    d'une URL (ex. trafic, age du domaine), elles sont mises a 0 ici."""
    lex = extract_lexical_features(url)
    row = {}
    for c in meta["numeric_cols"]:
        row[c] = 0.0  # valeur neutre, pas d'invention de donnees externes
    for c in meta["lexical_features"]:
        row[c] = float(lex.get(c, 0))
    return pd.DataFrame([[row[c] for c in meta["feature_names"]]],
                        columns=meta["feature_names"])


st.title("Detecteur de phishing : K-NN + K-Means")
st.write("Saisissez une URL pour estimer si elle est legitime ou suspecte. "
         "L'application n'ouvre jamais l'URL, elle analyse uniquement son texte.")

bundle = load_models()
if bundle is None:
    st.warning("Modeles non trouves. Lancez d'abord : python train.py (avec data/dataset.csv).")
    st.stop()

meta = bundle["meta"]

url = st.text_input("URL a analyser", placeholder="ex. https://example.com/login")
if st.button("Analyser"):
    if not url.strip():
        st.warning("Veuillez saisir une URL.")
    else:
        X = build_vector(url, meta)
        Xs = bundle["scaler"].transform(X)
        pred_knn = int(bundle["knn"].predict(Xs)[0])
        d = bundle["kmeans"].transform(Xs)
        import numpy as np
        pred_h = int(bundle["hybrid"].predict(np.hstack([Xs, d]))[0])
        cluster = int(bundle["kmeans"].predict(Xs)[0])
        label = lambda p: "Suspecte (phishing probable)" if p == 1 else "Legitime probable"
        st.subheader("Resultat")
        st.write(f"K-NN seul : **{label(pred_knn)}**")
        st.write(f"Hybride (K-Means + K-NN) : **{label(pred_h)}**")
        st.caption(f"URL rattachee au cluster K-Means n°{cluster} (regroupement non supervise).")
        if meta["numeric_cols"]:
            st.caption("Note : le dataset contient des colonnes non derivables d'une URL seule "
                       "(mises a 0 pour la prediction). La prediction sur URL libre est donc "
                       "indicative ; l'evaluation fiable est celle du tableau ci-dessous.")

st.header("Statistiques du dataset")
if os.path.exists(DATA_PATH):
    try:
        df = pd.read_csv(DATA_PATH, nrows=5000)
        st.write(f"Apercu (5 premieres lignes, {len(df)} lignes chargees max 5000) :")
        st.dataframe(df.head())
    except Exception as e:
        st.write(f"Lecture impossible : {e}")
else:
    st.write("Dataset non present dans data/dataset.csv.")

st.header("Performances (jeu de test, memes donnees pour les deux modeles)")
col1, col2 = st.columns(2)
with col1:
    st.subheader("K-NN seul")
    st.json(meta["metrics_knn"])
with col2:
    st.subheader("Hybride")
    st.json(meta["metrics_hybrid"])

st.subheader("Comparaison F1 / precision / rappel")
metrics = ["precision", "recall", "f1", "accuracy"]
fig, ax = plt.subplots()
x = range(len(metrics))
ax.bar([i - 0.2 for i in x], [meta["metrics_knn"][m] for m in metrics], width=0.4, label="K-NN")
ax.bar([i + 0.2 for i in x], [meta["metrics_hybrid"][m] for m in metrics], width=0.4, label="Hybride")
ax.set_xticks(list(x))
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1)
ax.legend()
st.pyplot(fig)

st.subheader("Matrices de confusion [TN FP / FN TP]")
c1, c2 = st.columns(2)
with c1:
    st.write("K-NN :")
    st.write(meta["confusion_knn"])
with c2:
    st.write("Hybride :")
    st.write(meta["confusion_hybrid"])
