"""Entrainement K-NN + K-Means (hybride) pour la detection de phishing.
Usage : python train.py
Attend le dataset reel dans data/dataset.csv (fourni par l'etudiant).
"""
import json
import os
import re
import sys
from urllib.parse import urlparse

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "data", "dataset.csv")
PHISH_PATH = os.path.join(BASE, "data", "phishtank_2026.csv")
LEGIT_PATH = os.path.join(BASE, "data", "legitimes.csv")
MODELS_DIR = os.path.join(BASE, "models")

# Echantillonnage documente : le flux PhishTank ne fournit que du phishing.
# On garde toutes les URL legitimes (pages d'accueil, data/legitimes.csv)
# et on tire au sort 3x plus d'URL de phishing (seed fixe -> reproductible).
PHISH_SAMPLE_RATIO = 3
SAMPLE_SEED = 42


def build_dataset_if_needed():
    """Construit data/dataset.csv a partir des deux sources si absent."""
    if os.path.exists(DATA_PATH):
        return
    if not (os.path.exists(PHISH_PATH) and os.path.exists(LEGIT_PATH)):
        print(f"[INFO] Dataset introuvable : {DATA_PATH}")
        print("       Placez votre fichier (nomme dataset.csv) dans data/ puis relancez : python train.py")
        sys.exit(1)
    ph = pd.read_csv(PHISH_PATH)
    lg = pd.read_csv(LEGIT_PATH)
    ph = ph.drop_duplicates(subset=[ph.columns[0]])
    n_ph = min(len(ph), PHISH_SAMPLE_RATIO * len(lg))
    ph_s = ph.sample(n=n_ph, random_state=SAMPLE_SEED)
    combo = pd.concat([ph_s, lg], ignore_index=True)
    combo.to_csv(DATA_PATH, index=False)
    print(f"[INFO] dataset.csv construit : {len(ph_s)} phishing (echantillon "
          f"aleatoire seed={SAMPLE_SEED} sur {len(ph)}) + {len(lg)} legitimes.")


def extract_lexical_features(url: str) -> dict:
    """Quelques caracteristiques simples extraites d'une URL (sans la visiter)."""
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


def detect_columns(df: pd.DataFrame):
    """Detecte la colonne URL et la colonne etiquette."""
    cols_lower = {c.lower().strip(): c for c in df.columns}
    url_col = None
    for cand in ["url", "urls", "lien", "link", "website", "domain"]:
        if cand in cols_lower:
            url_col = cols_lower[cand]
            break
    label_col = None
    for cand in ["label", "labels", "class", "classe", "target", "status",
                 "phishing", "result", "type", "verdict", "legitimate"]:
        if cand in cols_lower:
            label_col = cols_lower[cand]
            break
    # Dernier recours : derniere colonne comme etiquette
    if label_col is None and len(df.columns) >= 2:
        label_col = df.columns[-1]
    return url_col, label_col


def normalise_label(series: pd.Series) -> pd.Series:
    """Convertit les etiquettes en 0 (legitime) / 1 (phishing)."""
    s = series.copy()
    if s.dtype == object or str(s.dtype) == "string":
        low = s.astype(str).str.lower().str.strip()
        mapping_phish = {"phishing", "phish", "1", "true", "malicious",
                         "bad", "suspect", "suspecte", "fraudulent", "yes"}
        mapping_legit = {"legitimate", "legit", "0", "false", "benign",
                         "good", "legitime", "no", "safe"}
        out = low.map(lambda v: 1 if v in mapping_phish else (0 if v in mapping_legit else None))
        if out.isna().any():
            # Bascule generique : modalite la plus suspecte -> 1 si inconnu, on demande un mapping
            print("[WARN] Etiquettes non reconnues :", low.unique()[:10])
            print("       Adaptez normalise_label() au vocabulaire reel du dataset.")
            out = out.fillna(0).astype(int)
        return out.astype(int)
    return (s != 0).astype(int)


def main():
    build_dataset_if_needed()

    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Dataset : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    print("[INFO] Colonnes :", list(df.columns))

    url_col, label_col = detect_columns(df)
    print(f"[INFO] Colonne URL detectee : {url_col}")
    print(f"[INFO] Colonne etiquette detectee : {label_col}")
    if label_col is None:
        print("[ERREUR] Aucune colonne d'etiquette trouvee. Renommez-la (label/type/status) et relancez.")
        sys.exit(1)

    y = normalise_label(df[label_col])
    print("[INFO] Repartition des classes (0=legitime, 1=phishing) :")
    print(y.value_counts().to_string())

    # Caracteristiques : colonnes numeriques existantes + features lexicales si URL dispo
    numeric_cols = [c for c in df.columns if c != label_col and pd.api.types.is_numeric_dtype(df[c])]
    # Exclure une eventuelle colonne URL numerique par erreur
    if url_col in numeric_cols:
        numeric_cols.remove(url_col)

    use_lexical = url_col is not None
    if not numeric_cols and not use_lexical:
        print("[ERREUR] Ni colonne URL ni colonnes numeriques exploitables.")
        sys.exit(1)

    X_parts = []
    if numeric_cols:
        print(f"[INFO] Colonnes numeriques utilisees ({len(numeric_cols)}) : {numeric_cols}")
        X_parts.append(df[numeric_cols].fillna(0).astype(float))
    feat_names_lex = []
    if use_lexical:
        lex = df[url_col].apply(extract_lexical_features).apply(pd.Series)
        feat_names_lex = list(lex.columns)
        print(f"[INFO] Features lexicales extraites des URL : {feat_names_lex}")
        X_parts.append(lex)
    X = pd.concat(X_parts, axis=1)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)  # fit sur train uniquement (pas de fuite)
    X_test_s = scaler.transform(X_test)

    # --- K-NN seul ---
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_s, y_train)
    pred_knn = knn.predict(X_test_s)

    # --- K-Means (non supervise, k=2 : legitime vs phishing) ---
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(X_train_s)

    # --- Hybride : distances aux centres K-Means comme features supplementaires ---
    d_train = kmeans.transform(X_train_s)  # (n, 2)
    d_test = kmeans.transform(X_test_s)
    import numpy as np
    Xh_train = np.hstack([X_train_s, d_train])
    Xh_test = np.hstack([X_test_s, d_test])
    knn_h = KNeighborsClassifier(n_neighbors=5)
    knn_h.fit(Xh_train, y_train)
    pred_h = knn_h.predict(Xh_test)

    def scores(y_true, y_pred):
        return {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        }

    m_knn = scores(y_test, pred_knn)
    m_h = scores(y_test, pred_h)
    print("\n=== K-NN seul ===")
    print(m_knn)
    print("Matrice de confusion [TN FP / FN TP] :")
    print(confusion_matrix(y_test, pred_knn).tolist())
    print("\n=== Hybride K-Means + K-NN ===")
    print(m_h)
    print("Matrice de confusion [TN FP / FN TP] :")
    print(confusion_matrix(y_test, pred_h).tolist())

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(knn, os.path.join(MODELS_DIR, "knn.pkl"))
    joblib.dump(kmeans, os.path.join(MODELS_DIR, "kmeans.pkl"))
    joblib.dump(knn_h, os.path.join(MODELS_DIR, "knn_hybrid.pkl"))
    meta = {
        "label_col": label_col,
        "url_col": url_col,
        "numeric_cols": numeric_cols,
        "lexical_features": feat_names_lex,
        "feature_names": feature_names,
        "metrics_knn": m_knn,
        "metrics_hybrid": m_h,
        "confusion_knn": confusion_matrix(y_test, pred_knn).tolist(),
        "confusion_hybrid": confusion_matrix(y_test, pred_h).tolist(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    with open(os.path.join(MODELS_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Modeles enregistres dans {MODELS_DIR}/ (scaler, knn, kmeans, knn_hybrid, meta.json)")


if __name__ == "__main__":
    main()
