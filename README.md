# FireCast Tunisie

**Prédiction de l'occurrence d'incendies de végétation en Tunisie à partir de détections satellitaires (NASA FIRMS) et de données météorologiques (NASA POWER).**

Projet de Machine Learning, LSI 3. Auteurs : [Prénom Nom] et [Prénom Nom du binôme].

## 1. Problème

Question posée : *y aura-t-il au moins une détection satellitaire de feu demain dans cette cellule de 50 km ?*

- Type de tâche : apprentissage **supervisé**, **classification binaire** (`fire_tomorrow` = 1 ou 0).
- Unité d'observation : une cellule de grille (0,5° soit environ 50 km) un jour donné. Il y a 94 cellules couvrant la Tunisie.
- Les étiquettes proviennent des détections satellitaires, les variables explicatives de la météo, de l'historique de feux, de la saison et de la position.

## 2. Données

| Source | Contenu | Utilisation |
|---|---|---|
| NASA FIRMS (VIIRS SNPP, 375 m) | Détections de feux 2013-2024 | Étiquettes et historique de feux |
| NASA POWER | Température, humidité, vent, pluie quotidiens par cellule | Variables météo |

**Nettoyage des détections** (167 763 → 16 684) :
1. Conservation des feux de végétation uniquement (`type = 0`) : 50 138.
2. Suppression des détections de faible confiance : 44 079.
3. Conservation des points situés en Tunisie (frontières Natural Earth) : 22 853.
4. Suppression des sources fixes (même point de 1 km détecté sur 20 jours différents ou plus, par exemple torchères ou sites industriels) : 16 684.

**Jeu de données final** : `data/processed/firecast_dataset.csv.gz`, 409 182 lignes (94 cellules × 4 353 jours, du 2013-01-30 au 2024-12-30), 24 colonnes, 1,46 % de jours positifs.
Le fichier se lit directement avec `pandas.read_csv("firecast_dataset.csv.gz")`.

**Règle anti-fuite** : une ligne du jour D n'utilise que des informations du jour D ou antérieures. Seule l'étiquette regarde le jour D+1.

## 3. Méthodologie

- **Exploration** : déséquilibre des classes, saisonnalité, boîtes à moustaches, corrélations, valeurs aberrantes (la règle IQR est trompeuse pour la pluie, variable à zéros dominants : transformation `log1p` plutôt que suppression).
- **Découpage temporel** : entraînement 2013-2021 (306 252 lignes), test 2022-2024 (102 930 lignes). Aucun mélange aléatoire.
- **Sélection de variables** : filtre de corrélation (> 0,9), importance Random Forest et information mutuelle, calculés sur l'entraînement seulement. 13 variables retenues.
- **Modèles** : Régression logistique et Random Forest, avec `class_weight` pour le déséquilibre (environ 64 jours sans feu pour 1 avec feu).
- **Réglage** : `GridSearchCV` avec `TimeSeriesSplit(n_splits=4)`, score = PR-AUC (précision moyenne).
- **Seuil de décision** : choisi sur des prédictions hors échantillon de l'entraînement, jamais sur le test.
- **Références** : taux mensuel, présence d'un feu dans les 7 derniers jours, nombre de feux des 30 derniers jours.
- **Ablation** : retrait de groupes de variables (historique de feux, position, météo, saison).

## 4. Résultats (années de test 2022-2024)

| Modèle | PR-AUC | ROC-AUC | Précision | Rappel | F1 |
|---|---|---|---|---|---|
| Modèle aléatoire | 0,012 | 0,500 | n/a | n/a | n/a |
| Référence A : taux mensuel | 0,019 | 0,645 | n/a | n/a | n/a |
| Référence B' : nombre de feux sur 30 jours | 0,112 | 0,793 | n/a | n/a | n/a |
| Régression Logistique (seuil 0,92) | 0,161 | 0,852 | 0,217 | 0,269 | 0,241 |
| **Random Forest (seuil 0,90)** | **0,179** | **0,874** | 0,251 | 0,249 | 0,250 |

Hyperparamètres retenus : Régression Logistique `C = 0,001` ; Random Forest `max_depth = 12`, `min_samples_leaf = 100`, 100 arbres.

**Conclusions**
- Les modèles classent les jours à risque environ 14 fois mieux que le hasard.
- L'ablation montre que la majeure partie de la performance vient de l'historique récent de feux et de la position. Dans ce montage, la météo quotidienne apporte peu.

**Limites** : météo grossière (cellules d'environ 50 km), pas de données de végétation ni de sécheresse du combustible, détections satellitaires incomplètes (nuages, passages) et incluant des brûlages agricoles, performance en baisse en 2023-2024 sans cause identifiée. Les scores du modèle ne sont pas des probabilités calibrées.

## 5. Structure du dépôt

```
FireCast/
├── README.md
├── requirements.txt
├── data/processed/
│   ├── firecast_dataset.csv.gz     <- jeu de données final
│   ├── fires_tunisia_clean.csv     <- détections nettoyées
│   ├── grid_cells.csv              <- les 94 cellules
│   ├── train.csv.gz, test.csv.gz   <- découpage temporel
│   └── features.json, selected_features.json
├── src/
│   ├── demo.py                     <- démonstration rapide (voir section 6)
│   ├── download_firms.py, clean_fires.py, build_grid.py
│   ├── download_power.py, build_dataset.py, check_label.py
│   ├── explore_dataset.py, check_hotspots.py, check_rain.py
│   ├── prepare_data.py, select_features.py, finalize_features.py
│   ├── train_models.py, evaluate_models.py, ablation.py, risk_map.py
│   └── config_example.py
└── results/
    ├── figures/                    <- graphiques
    ├── tables/                     <- tableaux de résultats
    └── best_params.json, thresholds.json, feature_ranking.csv
```



## 6. Reproduire toutes les étapes

**À partir du jeu de données final** (aucune clé nécessaire) :

```bash
python src/prepare_data.py       # découpage temporel, log1p, filtre de corrélation
python src/select_features.py    # classement des variables
python src/finalize_features.py  # 13 variables retenues
python src/train_models.py       # références + 2 modèles, GridSearchCV (CV = 4), quelques minutes
python src/evaluate_models.py    # seuils, matrices de confusion, courbes PR
python src/ablation.py           # étude d'ablation
```


Exécuter dans l'ordre : `download_firms.py`, `clean_fires.py`, `build_grid.py`, `download_power.py`, `build_dataset.py`, puis `check_label.py` pour vérifier les étiquettes.

Les exécutions se font depuis la racine du projet. Les fichiers bruts (`data/raw/`) et les modèles enregistrés (`results/models/`) ne sont pas versionnés car volumineux.
