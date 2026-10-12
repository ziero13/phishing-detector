# Rapport de projet — Détection hybride des URL de phishing (K-NN + K-Means)

| Champ | À compléter |
|---|---|
| Membres du groupe | … / … / … |
| Université | … |
| Filière | … |
| Année académique | … |
| Cours | Intelligence Artificielle |
| Sujet | Détection hybride des URL de phishing avec K-NN et K-Means |

---

## 1. Objectif

Construire une application Python qui analyse une URL et prédit si elle est
**légitime (0)** ou **phishing (1)**, en combinant un classifieur **K-NN** et un
clustering **K-Means**, avec une interface web **Streamlit** en français.

Dépôt GitHub : https://github.com/ziero13/phishing-detector

## 2. Données utilisées

- **Source fournie** : `data/phishtank_2026.csv` — 64 280 URL PhishTank, toutes
  étiquetées phishing (`Label=1`).
- **Problème rencontré** : aucune URL légitime dans le fichier, donc aucun
  classifieur binaire ne peut s'entraîner dessus tel quel.
- **Solution retenue** : ajout de 574 pages d'accueil de domaines publics
  notoires (`https://domaine/`, `https://www.domaine/`), listées explicitement
  dans `data/legitimes.csv`, étiquetées légitimes (`Label=0`). Aucune URL visitée.
- **Jeu final** : `train.py` construit `data/dataset.csv` = 574 légitimes +
  1 722 phishing tirées au sort (seed 42, ratio 3:1), soit 2 296 URL.

> 📷 **[INSÉRER CAPTURE 1 — aperçu de `data/dataset.csv` ou `df.head()`]**

## 3. Caractéristiques extraites (features)

Chaque URL est convertie en 10 nombres, sans jamais l'ouvrir : longueur totale,
longueur de l'hôte, points, tirets, slashs, chiffres, caractères spéciaux
(`@ ? = % & _ ~`), présence de HTTPS, IP dans l'hôte, sous-domaines.
Principe : une URL de phishing est souvent longue, chargée en tirets,
chiffres et caractères spéciaux.

Voir `train.py` → `extract_lexical_features()` sur GitHub.

## 4. Modèle K-NN

- Features standardisées (`StandardScaler` ajusté **sur le train uniquement**).
- Split 80 % entraînement / 20 % test, stratifié, seed 42.
- `KNeighborsClassifier(k=5)` : une URL est classée selon le vote de ses
  5 voisines les plus proches.

## 5. Modèle K-Means

- `KMeans(k=2)` ajusté sur le train : regroupe les URL par ressemblance,
  **sans utiliser les étiquettes** (non supervisé).
- Sert à rattacher chaque URL à un cluster et à mesurer ses distances aux
  2 centres.

## 6. Approche hybride

Les 2 distances aux centres K-Means sont concaténées aux 10 features, puis un
second K-NN (k=5) est entraîné sur ces 12 valeurs. L'information de
regroupement complète le vote des voisins. Pas de fuite de données : scaler et
K-Means ajustés sur le train, appliqués tels quels au test.

Voir `train.py` sur GitHub.

> 📷 **[INSÉRER CAPTURE 2 — sortie console de `python train.py` (métriques)]**

## 7. Résultats (jeu de test : 460 URL)

| Modèle | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| K-NN seul | 0.9326 | 0.9486 | 0.9623 | 0.9554 |
| Hybride K-Means + K-NN | 0.9348 | 0.9513 | 0.9623 | 0.9568 |

- Matrices de confusion [TN FP / FN TP] : K-NN `[[97, 18], [13, 332]]`,
  hybride `[[98, 17], [13, 332]]`.
- Lecture : l'hybride corrige **1 faux positif**. Gain modeste mais réel :
  le signal de regroupement aide dans les cas limites.
- Scores de fiabilité des probabilités : Brier ≈ 0.045 (bien orientées) mais
  valeurs grossières (pas de 20 % avec k=5) → l'interface les présente comme
  un **score indicatif**, pas une probabilité calibrée.

## 8. Interface Streamlit

- Champ de saisie + bouton **Analyser** : catégorie prédite (légitime/suspecte),
  score de phishing en % (K-NN et hybride), barre de progression.
- Section **« Pourquoi ce résultat ? »** : signaux en clair (IP, pas de HTTPS,
  tirets, longueur…).
- Onglets de présentation : statistiques du dataset, métriques, graphique
  comparatif K-NN vs hybride, matrices de confusion.

> 📷 **[INSÉRER CAPTURE 3 — interface : analyse d'une URL suspecte (score + explication)]**

> 📷 **[INSÉRER CAPTURE 4 — interface : métriques et graphique comparatif]**

## 9. Lancement du projet

```text
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py                  # construit le dataset, entraîne, évalue
streamlit run app.py             # interface sur http://localhost:8501
```

Les modèles entraînés sont conservés dans `models/` : l'interface les charge
directement, **sans réentraînement** à chaque lancement.

## 10. Limites connues

- Les URL légitimes sont des pages d'accueil, pas un échantillon représentatif
  du trafic réel : la prédiction sur URL libre reste indicative.
- Quelques phishing sont hébergés sur des infrastructures légitimes
  (ex. `docs.google.com`) : seule la page d'accueil est classée légitime.
- Classes déséquilibrées (3:1) et score non calibré : assumés et documentés.

## 11. Conclusion

Application fonctionnelle et présentable : K-NN seul F1 = 0.9554, hybride
F1 = 0.9568. La combinaison K-Means + K-NN apporte un gain petit mais mesuré
et justifié, avec une interface qui explique chaque verdict.
