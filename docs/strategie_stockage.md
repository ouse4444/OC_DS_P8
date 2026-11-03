# Stratégie de stockage des données de production

Projet 8 : mise en production du modèle de scoring crédit de Prêt à Dépenser.

Ce document définit quelles données de production sont conservées, où, pourquoi, combien de temps et sous quelles protections. Il justifie l'adéquation de cette solution aux exigences du suivi (monitoring) du modèle.

## 1. Les exigences du suivi

Le suivi d'un modèle en production doit répondre à cinq questions. Chacune impose des données à conserver :

| Question de suivi | Indicateurs | Données nécessaires |
|---|---|---|
| Le service fonctionne-t-il ? | Volume de requêtes, taux d'erreur, disponibilité | Date, code de réponse de chaque appel, y compris les rejets et les erreurs |
| Le service répond-il assez vite ? | Temps de traitement (médiane, 95e centile) | Temps d'inférence du modèle et temps total de traitement de chaque appel |
| Les décisions sont-elles stables ? | Distribution des scores, taux de refus | Probabilité, décision et seuil de chaque prédiction |
| Les données reçues ressemblent-elles à celles de l'entraînement ? | Dérive des données (Evidently, PSI) | Les variables reçues pour chaque prédiction |
| Le modèle prédit-il toujours correctement ? | AUC, rappel, précision, une fois les défauts connus | Un identifiant de requête, pour relier plus tard chaque prédiction à la vraie issue du crédit |

S'y ajoutent deux exigences transverses : la **traçabilité** (savoir quel modèle a produit quelle décision) et la **protection des données personnelles** (RGPD).

## 2. Ce qui est stocké

Chaque appel à la route `/predict`, réussi ou non, produit **une ligne** dans la table `predictions`. La table est créée automatiquement par l'API à son démarrage (module `api/prediction_log.py`).

| Colonne | Type | Rôle dans le suivi |
|---|---|---|
| `id` | entier | Clé technique |
| `created_at` | date et heure (UTC), indexée | Suivi dans le temps, filtres par période |
| `request_id` | texte, unique | Identifiant renvoyé au client : relie la réponse et la ligne en base, puis la prédiction et la vraie issue |
| `model_name`, `model_version` | texte | Traçabilité : quel modèle a pris la décision |
| `status_code` | entier, indexé | 200 succès, 422 donnée invalide, 503 modèle indisponible, 500 erreur imprévue |
| `probability_default`, `decision`, `threshold` | nombre, texte, nombre | Distribution des scores, taux de refus |
| `inference_time_ms`, `total_time_ms` | nombres | Temps de réponse |
| `inputs` | JSONB | Variables reçues, pour l'analyse de dérive |
| `error` | texte | Cause d'un rejet ou type d'une erreur (sans détail interne) |

**Pourquoi enregistrer aussi les échecs ?** Un taux d'erreur ne se calcule qu'en conservant les erreurs. Les requêtes rejetées révèlent aussi des problèmes côté client (format, version du contrat d'interface).

**Pourquoi le format JSONB pour les entrées ?** Le modèle reçoit jusqu'à 199 variables, dont la plupart sont facultatives. Une colonne par variable rendrait la table rigide : toute évolution du modèle imposerait une modification de structure. Le format JSONB de PostgreSQL stocke l'ensemble des variables reçues dans une seule colonne, en format binaire, tout en restant interrogeable en SQL (par exemple `inputs->>'AMT_CREDIT'`).

**Double écriture.** Chaque événement est aussi écrit sous forme d'une ligne JSON dans les journaux de l'API, collectés automatiquement par Azure (Log Analytics). Si la base est indisponible, l'événement reste ainsi tracé dans les journaux.

## 3. Où et pourquoi : le choix de PostgreSQL sur Azure

**Solution retenue : Azure Database for PostgreSQL (serveur flexible), région France Central, base `monitoring`.**

| Option étudiée | Avantages | Limites | Décision |
|---|---|---|---|
| **PostgreSQL géré sur Azure** | Requêtes SQL (centiles, regroupements par période, jointures), format JSONB, accessible à la fois par l'API, le tableau de bord et le notebook d'analyse, sauvegardes automatiques, même région que l'API | Une ressource facturée en continu (environ 0,50 $ par jour en taille minimale) | **Retenue** |
| Journaux Azure seuls (Log Analytics) | Gratuit jusqu'à 5 Go par mois, déjà en place | Langage de requête spécifique (KQL), moins adapté aux analyses statistiques et au tableau de bord | Conservé en complément (double écriture) |
| Fichiers Parquet sur un stockage objet (Blob Storage) | Très peu coûteux pour de gros volumes, idéal pour l'analyse par lots | Pas de requêtes en temps réel ; écriture concurrente délicate depuis une API | Écarté à ce volume ; pertinent pour l'archivage à grande échelle |
| Base PostgreSQL locale (Docker) | Gratuite | Inaccessible depuis l'API déployée sur Azure : ne stocke pas les vraies données de production | Écartée |

**Adéquation aux exigences du suivi.**

- **Temps réel** : chaque prédiction est disponible en base dès la fin de la requête. Le tableau de bord la voit à sa prochaine actualisation (une minute au plus).
- **Analyse** : les centiles de latence sont calculés directement en SQL (`PERCENTILE_CONT`), les entrées JSONB sont relues par le notebook Evidently.
- **Indépendance des services** : l'API écrit, le tableau de bord et le notebook lisent. Aucun ne dépend du code des autres : la base est le point de rencontre.
- **Performance** : l'écriture est faite après l'envoi de la réponse au client (tâche de fond). Elle n'ajoute rien au temps de réponse, et une base indisponible ne fait jamais échouer une prédiction (comportement vérifié par des tests automatisés).

## 4. Volume et coût

| Élément | Estimation |
|---|---|
| Taille d'une ligne | Environ 5 Ko (dont l'essentiel pour les entrées JSONB, avec 199 variables) |
| 10 000 prédictions par jour pendant un an | Environ 18 Go |
| Capacité du serveur | 32 Go de stockage (taille minimale), extensible |
| Coût | Environ 15 $ par mois (calcul de taille minimale et stockage), sous une alerte budgétaire de 20 $ |

Au-delà de quelques dizaines de millions de lignes, la stratégie évoluerait : partitionnement de la table par mois, et archivage des mois anciens en fichiers Parquet sur un stockage objet, beaucoup moins coûteux.

## 5. Durée de conservation

| Données | Durée | Justification |
|---|---|---|
| Lignes détaillées (entrées, score, décision) | **13 mois**, puis suppression | Couvrir un cycle annuel complet (saisonnalité) et le délai nécessaire pour connaître l'issue des crédits et mesurer la performance réelle |
| Indicateurs agrégés (volumes, scores moyens, taux de refus, latences par jour) | Sans limite | Ne contiennent aucune donnée individuelle ; permettent les comparaisons de long terme |
| Rapports d'analyse de dérive | Sans limite (dans le dépôt) | Statistiques agrégées, sans donnée individuelle |

La suppression des lignes de plus de 13 mois prend la forme d'une requête planifiée (`DELETE FROM predictions WHERE created_at < now() - interval '13 months'`). Dans le cadre de ce projet, elle est définie mais pas encore planifiée : c'est une étape de mise en production réelle.

## 6. Protection des données

| Mesure | Mise en œuvre |
|---|---|
| Minimisation | Aucune donnée d'identification directe n'est envoyée ni stockée (pas de nom, d'adresse ni d'identifiant client) ; seules les variables utiles au modèle sont conservées |
| Localisation | Région France Central : les données restent dans l'Union européenne |
| Chiffrement en transit | Connexion obligatoirement chiffrée (TLS 1.3, `sslmode=require`) |
| Chiffrement au repos | Assuré par Azure sur le stockage de la base |
| Contrôle d'accès | Pare-feu : seuls les services Azure (l'API, le tableau de bord) et le poste d'analyse sont autorisés ; aucune autre adresse ne peut se connecter |
| Secrets | L'adresse de connexion, qui contient le mot de passe, est un secret de chaque application Azure et un fichier `.env` local non versionné ; elle n'apparaît ni dans le code, ni dans GitHub, ni dans les images Docker |
| Sauvegardes | Sauvegardes automatiques du serveur géré (7 jours de rétention par défaut) |
| Tableau de bord | Il compare la production à un profil agrégé de la référence (déciles et proportions) : aucune donnée d'entraînement individuelle n'est embarquée |

## 7. Limites et améliorations

- Les requêtes rejetées ne renvoient pas d'identifiant au client : elles sont stockées, mais ne peuvent pas être reliées à leur émetteur. Amélioration : renvoyer l'identifiant de requête dans toutes les réponses.
- La vraie issue des crédits n'est pas encore stockée : une table complémentaire, reliée par `request_id`, permettrait de suivre automatiquement la performance réelle du modèle.
- La suppression au-delà de 13 mois reste à planifier.
- Les accès en lecture (tableau de bord, analyse) utilisent le compte administrateur : un compte en lecture seule appliquerait plus strictement le principe du moindre privilège.

## 8. Preuves

Les captures d'écran du livrable 7 montrent :
1. la structure de la table `predictions`, créée par l'API ;
2. les dernières prédictions enregistrées, avec score, décision et temps de traitement ;
3. le décompte des appels par code de réponse ;
4. le tableau de bord de suivi, alimenté par cette table.
