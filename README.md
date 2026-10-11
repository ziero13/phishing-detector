# Détection hybride d'URL de phishing (K-NN + K-Means)

TP universitaire — cybersécurité. Application simple : classifier des URL
(légitime vs phishing) avec K-NN, regrouper avec K-Means, combiner les deux
(distances aux centres K-Means ajoutées comme features pour K-NN).

## Structure

```
phishing-detector/
├── data/
│   ├── phishtank_2026.csv  # source phishing (64280 URL, toutes Label=1)
│   ├── legitimes.csv       # 574 pages d'accueil de domaines publics (Label=0)
│   └── dataset.csv         # construit auto par train.py (1722 phishing + 574 legitimes)
├── models/        # modèles entraînés (générés par train.py)
├── app.py         # interface Streamlit (français)
├── train.py       # préparation, entraînement, évaluation
├── requirements.txt
└── README.md
```

## Données

- **Phishing** : fichier fourni par l'étudiant (`phishtank_2026.csv`,
  64280 URL étiquetées 1, 64266 uniques). Problème constaté : il ne
  contient **aucune URL légitime**, donc aucun classifieur binaire ne
  peut s'entraîner dessus tel quel.
- **Légitimes** : 574 pages d'accueil (`https://domaine/` et
  `https://www.domaine/`) de 287 domaines publics notoires, listées
  explicitement dans `data/legitimes.csv`. Aucune URL visitée.
- **Échantillonnage** : `train.py` construit `dataset.csv` = 574
  légitimes + 1722 phishing tirés au sort (seed 42, ratio 3:1).
- Nuance à signaler au jury : quelques URL de phishing sont hébergées
  sur des infrastructures légitimes (ex. `docs.google.com`), seule la
  page d'accueil du domaine est classée légitime — les features
  lexicales (longueur, caractères spéciaux) les distinguent.

## Installation

```bash
python -m venv .venv
# Linux / macOS :
source .venv/bin/activate
# Windows (PowerShell) :
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Utilisation

1. Placer le dataset fourni dans `data/dataset.csv`.
2. Entraîner et évaluer :
   ```bash
   python train.py
   ```
   Affiche : colonnes détectées, répartition des classes, métriques
   (accuracy, precision, recall, F1) pour K-NN seul et hybride,
   matrices de confusion. Modèles sauvegardés dans `models/`.
3. Lancer l'interface :
   ```bash
   streamlit run app.py
   ```

## Algorithmes

- **K-NN (k=5)** : classifieur supervisé sur features standardisées.
- **K-Means (k=2)** : clustering non supervisé (regroupement légitime/phishing).
- **Hybride** : distances aux 2 centres K-Means concaténées aux features,
  puis K-NN (k=5) dessus. Comparaison sur le même jeu de test
  (split 80/20 stratifié, scaler et K-Means ajustés sur train uniquement).

## Résultats (jeu de test 20 %, 460 URL — sortie réelle de train.py)

| Modèle | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| K-NN seul | 0.9326 | 0.9486 | 0.9623 | 0.9554 |
| Hybride K-Means + K-NN | 0.9348 | 0.9513 | 0.9623 | 0.9568 |

Matrices de confusion [TN FP / FN TP] : K-NN `[[97, 18], [13, 332]]`,
hybride `[[98, 17], [13, 332]]`. L'hybride corrige 1 faux positif ;
gain modeste mais réel et expliqué : les distances aux centres
apportent une information de regroupement complémentaire.

## Notes

- L'app n'ouvre jamais les URL saisies (analyse lexicale locale uniquement).
- Si le dataset contient des colonnes numériques non dérivables d'une URL
  seule, la prédiction interactive les met à 0 (indiquée comme indicative
  dans l'interface) ; l'évaluation fiable reste celle sur le jeu de test.
