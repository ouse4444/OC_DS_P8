# Rapport d'analyse et d'optimisation des performances de l'API

Projet 8 : mise en production du modèle de scoring crédit de Prêt à Dépenser.

## 1. Objectif et démarche

Le département Crédit Express attend une réponse en quasi temps réel : la **latence** de l'API est le critère principal. L'optimisation a suivi quatre étapes :

1. **Mesurer** : découper le temps d'une requête étape par étape (profilage), sur le vrai modèle et de vrais clients.
2. **Optimiser** uniquement ce que la mesure désigne comme coûteux.
3. **Prouver l'absence de régression** : mêmes prédictions, même AUC.
4. **Déployer** par le pipeline CI/CD, puis **mesurer en production** sur Azure.

Outils utilisés : `time.perf_counter` pour chronométrer chaque étape, le script `scripts/benchmark_inference.py` (comparaison avant et après sur 500 clients du jeu de test, démarrage mesuré dans 5 processus Python neufs), les centiles calculés en SQL (`PERCENTILE_CONT`) sur la table des prédictions de production, et le pipeline GitHub Actions pour la taille de l'image.

## 2. Diagnostic : où passe le temps d'une requête ?

Mesures sur la première version de l'API (médiane sur 500 clients, Mac à puce Apple) :

| Étape | Temps médian | Constat |
|---|---|---|
| Validation des entrées (Pydantic) | 0,06 ms | Négligeable |
| Contrôle des variables complémentaires | 0,18 ms | Négligeable |
| **Construction de la ligne de données (pandas)** | **5,35 ms** | **Étape la plus coûteuse, devant le modèle** |
| Pipeline du modèle (variables construites, imputation, encodage, LightGBM) | 2,94 ms | Dont LightGBM seul : moins de 1 ms |
| **Requête complète** | **8,86 ms** | |

Deux problèmes ont été identifiés :

1. **La mise en forme des données coûtait plus que le modèle.** La ligne de 199 variables était d'abord construite, puis ses 188 colonnes numériques étaient converties après coup. Sur Azure, avec un seul processeur partagé, ce surcoût atteignait environ 17 ms par requête et faisait attendre les requêtes simultanées.
2. **L'API importait toute la bibliothèque MLflow** uniquement pour lire le modèle. MLflow sert au suivi des expériences et au registre ; il alourdissait l'image Docker et le démarrage, ce qui allonge le démarrage à froid sur Azure (mise en veille quand l'API n'est pas utilisée).

## 3. Optimisations réalisées

| # | Optimisation | Principe | Fichier |
|---|---|---|---|
| 1 | Construction directe de la ligne | Chaque colonne est créée directement avec son type final (décimal ou objet), au lieu d'être convertie ensuite | `api/model_service.py`, méthode `to_dataframe` |
| 2 | Chargement du modèle sans MLflow | Le fichier `model.pkl` exporté par MLflow (format cloudpickle, indiqué dans `model/MLmodel`) est lu directement ; MLflow est retiré des dépendances de l'image de production et reste utilisé dans l'environnement d'analyse | `api/model_service.py`, `requirements.txt` |

Les versions des bibliothèques du modèle restent figées sur celles de l'environnement d'export (scikit-learn 1.8.0, LightGBM 4.6.0, pandas 2.3.3, NumPy 2.4.6, SciPy 1.17.1, cloudpickle 3.1.2).

## 4. Résultats

### 4.1 Mesures locales (script de benchmark, 500 clients du jeu de test)

| Mesure | Avant | Après | Gain |
|---|---|---|---|
| Construction de la ligne (médiane) | 5,35 ms | 0,91 ms | **-83 %** |
| Requête complète (médiane) | 8,86 ms | 3,90 ms | **-56 %** |
| Requête complète (95e centile) | 9,06 ms | 3,99 ms | **-56 %** |
| Démarrage : import et chargement du modèle | 0,98 s | 0,63 s | **-36 %** |

Après optimisation, le pipeline du modèle (2,9 ms) représente l'essentiel du temps de la requête : il n'y a plus de gaspillage autour du modèle.

### 4.2 Mesures en production (Azure Container Apps, table des prédictions PostgreSQL)

Même trafic simulé (clients du jeu de test, 4 requêtes simultanées), avant et après le déploiement :

| Mesure | Avant (843 prédictions) | Après (500 prédictions) | Gain |
|---|---|---|---|
| Traitement total dans l'API (médiane) | 90,1 ms | 30,4 ms | **-66 %** |
| Traitement total dans l'API (95e centile) | 184,4 ms | 60,0 ms | **-67 %** |
| Inférence du modèle (médiane) | 29,9 ms | 22,6 ms | -24 % |

Le gain est plus fort en production qu'en local : sur un processeur partagé, la conversion coûteuse ralentissait aussi les requêtes simultanées. En la supprimant, le processeur est libéré pour toutes les requêtes, y compris pour le calcul du modèle, qui n'a pourtant pas été modifié.

### 4.3 Image Docker de l'API

| Mesure | Avant | Après | Gain |
|---|---|---|---|
| Taille sur disque | 1,27 Go | 744 Mo | **-41 %** |
| Taille compressée (téléchargée par Azure) | 289 Mo | 169 Mo | **-42 %** |

Une image plus légère se télécharge plus vite : le démarrage à froid de l'API sur Azure est raccourci d'autant.

## 5. Absence de régression

| Contrôle | Résultat |
|---|---|
| Écart maximal de probabilité, client par client (500 clients) | **0,0** : prédictions strictement identiques |
| AUC sur l'ensemble du jeu de test du Projet 6 (61 503 clients) | **0,7855 avant, 0,7855 après** |
| Tests automatisés | 71 tests, dont deux nouveaux : l'API n'importe plus MLflow ; la ligne construite a exactement les 199 colonnes, dans l'ordre et avec le bon type |

## 6. Pistes étudiées et non retenues

| Piste | Raison du choix |
|---|---|
| Préchauffage du modèle au démarrage | La première requête n'est pas sensiblement plus lente que les suivantes (environ 32 ms contre 28 ms) : gain négligeable |
| Conversion du modèle au format ONNX | Le LightGBM ne prend que 0,8 ms ; le gain serait minime, pour un risque d'écart de prédiction et une complexité accrue (la classe de construction des variables du Projet 6 n'est pas convertible directement) |
| Appel direct du moteur LightGBM, sans l'enveloppe scikit-learn | Gain inférieur à 1 ms, au prix d'un contournement du pipeline : la cohérence entre entraînement et production prime |

## 7. Perspectives

- **Supprimer le démarrage à froid** en gardant une copie de l'API toujours active (`--min-replicas 1`), si le métier exige une réponse immédiate à toute heure : la facturation devient alors permanente.
- **Augmenter la capacité** sous forte charge : plusieurs processus uvicorn par conteneur, ou plusieurs copies de l'API (mise à l'échelle horizontale automatique de Container Apps).
- **Alléger encore l'image** avec une construction en plusieurs étapes (multi-stage) et une image de base plus petite.
- **Mesurer en continu** : le tableau de bord suit déjà la latence en production (onglet Temps de réponse) ; un seuil d'alerte sur le 95e centile (par exemple 300 ms) permettrait de détecter toute régression future.
