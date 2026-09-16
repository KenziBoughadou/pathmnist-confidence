# PathMNIST Confidence

**Un modèle plus performant produit-il aussi des probabilités plus fiables ?**
Cette étude compare une régression logistique et un petit CNN sur des images
histologiques, puis mesure l'effet d'une calibration du CNN et de l'abstention
sur les exemples les moins confiants.

C'est un **projet académique personnel de classification, calibration et
évaluation de modèles**. Le CNN est entraîné depuis une initialisation aléatoire.
La question porte autant sur les erreurs confiantes que sur le nombre de bonnes
prédictions. **Aucun usage diagnostique ou clinique.**

| L'expérience | Choix retenu |
|---|---|
| Données | PathMNIST, images couleur 28 × 28, neuf types de tissus et de contenu |
| Apprentissage | 89 996 images, partition officielle |
| Sélection / calibration | Deux moitiés stratifiées distinctes de 5 002 images |
| Test final | 7 180 images, partition officielle issue d'un autre centre |
| États comparés | Régression logistique, CNN brut, même CNN calibré |
| Répétitions | Une baseline linéaire ; trois graines CNN, quinze époques chacune |
| Évaluation | Exactitude, macro-F1, NLL, Brier, ECE et risque-couverture |
| Exploration | Streamlit à partir des prédictions enregistrées |

## Résultats observés

Évaluation sur les **7 180 images du test officiel**, après tous les entraînements
et ajustements de température. Le CNN est résumé par sa moyenne ± écart-type
d'échantillon sur trois graines ; la régression logistique n'a qu'une exécution.

| État | Exactitude (%) | Macro-F1 | NLL ↓ | Brier ↓ | ECE ↓ |
|---|---:|---:|---:|---:|---:|
| Régression logistique | 52,84 | 0,4353 | 1,2444 | 0,5770 | 0,0346 |
| CNN brut | 77,47 ± 2,12 | 0,7077 ± 0,0191 | 0,8002 ± 0,0946 | 0,3342 ± 0,0297 | 0,0362 ± 0,0119 |
| Même CNN calibré | 77,47 ± 2,12 | 0,7077 ± 0,0191 | 0,8164 ± 0,1037 | 0,3346 ± 0,0310 | 0,0378 ± 0,0192 |

**La calibration n'améliore pas systématiquement les résultats du test.**
Les températures des graines 0, 1 et 2 valent respectivement 0,9578, 0,9339 et
0,9278. Elles réduisent légèrement la NLL de calibration, mais augmentent la NLL
du test pour les trois graines. L'ECE diminue pour les graines 0 et 1, et augmente
pour la graine 2. Le Brier diminue uniquement pour la graine 1. L'exactitude et le
macro-F1 restent strictement inchangés, comme attendu avec une température positive.

Le CNN classe mieux que cette baseline, sans être meilleur selon toutes les
mesures de calibration. **La régression logistique atteint sa limite de 300
itérations sans convergence** : son résultat ne représente donc pas une baseline
linéaire pleinement optimisée. Son ECE relativement faible, malgré une exactitude
modeste, illustre aussi pourquoi ce score seul ne suffit pas à comparer les modèles.

Le test vient d'un centre distinct du développement. C'est un contexte pertinent
pour observer ces écarts, mais cette expérience ne permet pas d'en attribuer la
cause au centre, ni de démontrer une robustesse au changement de distribution.

### Abstention : un gain conditionnel, des erreurs persistantes

Pour la **graine 0**, choisie à l'avance pour les illustrations :

| Couverture | Exactitude CNN brut (%) | Exactitude CNN calibré (%) |
|---|---:|---:|
| 100 % | 78,12 | 78,12 |
| 90 % | 82,08 | 82,06 |
| 75 % | 86,78 | 86,72 |
| 50 % | 89,97 | 90,06 |

![Risque selon la couverture, graine 0](results/study/report/risk-coverage-seed-0.png)

Écarter la moitié des images augmente l'exactitude sur les images restantes,
mais laisse environ 10 % d'erreurs. La courbe n'est pas monotone : certains
exemples erronés se classent parmi les plus confiants. La calibration modifie
peu ce classement. L'AURC moyenne passe de **0,09918 ± 0,02034** à
**0,09876 ± 0,02036** ; cette petite baisse ne signifie pas une amélioration à
chaque couverture, ni une garantie de risque futur.

### Erreurs à haute confiance

![Dix erreurs les plus confiantes du CNN, graine 0](results/study/report/cnn-seed-0-top-errors.png)

Les images sont sélectionnées automatiquement, sans filtrage visuel. Pour les
graines 0, 1 et 2, le nombre d'erreurs de confiance ≥ 0,9 passe respectivement de
**334 à 355**, **120 à 145** et **250 à 308** après calibration. Les températures
inférieures à 1 rendent ici les distributions plus concentrées, y compris sur
des prédictions incorrectes.

Pour la graine 0, la paire débris → muscle lisse représente 128 de ces erreurs
avant calibration, puis 129 après ; mucus → fond apparaît aussi parmi les paires
fréquentes des trois graines. Ce sont des comptages du modèle, sans interprétation
histopathologique indépendante.

### Coût et traçabilité

Le CNN compte **106 089 paramètres**, contre **21 177** pour la baseline.
L'entraînement CNN prend **713,1 ± 8,8 s** par graine, sélection et sauvegardes
comprises ; la régression logistique prend **241,2 s**, conversion des pixels
comprise. Ces périmètres diffèrent et le CPU est partagé. Le téléchargement,
la calibration et l'évaluation finale sont exclus de ces durées.

Le [rapport complet](results/study/report/README.md) présente chaque graine,
les diagrammes de fiabilité, courbes, matrices de confusion et erreurs. Les
[mesures non arrondies](results/study/report/runs.csv) et
[écarts calibré − brut](results/study/report/paired-differences.csv) permettent
de vérifier la synthèse. Une [première tentative interrompue](results/README.md)
est conservée ; elle n'avait pas atteint l'évaluation du test. Aucun entraînement
n'a été relancé en fonction de sa performance de test.

## Méthode

### Sélectionner, calibrer, puis évaluer

```text
Entraînement officiel ────────── apprentissage des poids

Validation officielle ┬──────── sélection du checkpoint : 5 002 images
                      └──────── ajustement de T uniquement : 5 002 images

Test officiel ───────────────── évaluation finale après sélection et calibration
```

La séparation de la validation est stratifiée, déterministe et enregistrée avant
les expériences. Les images utilisées pour choisir le checkpoint ne servent pas
à ajuster la température. Les hyperparamètres sont fixés, sans recherche sur le test.
Les règles numériques sont détaillées dans le [protocole](docs/PROTOCOL.md) et
leur [configuration](protocol.json).

**La régression logistique** reçoit les 2 352 pixels aplatis, divisés par 255.
Elle fournit une baseline linéaire scikit-learn, avec régularisation L2, `C=1`,
solveur L-BFGS et au plus 300 itérations. Ses éventuels avertissements de convergence
restent dans les résultats.

**Le CNN** reçoit les images dans leur structure spatiale. Il comporte deux blocs
convolution–ReLU–max-pooling, avec 16 puis 32 canaux, suivis d'une couche dense
de 64 neurones et de neuf logits. La boucle PyTorch est explicite : calcul de
la perte, remise à zéro des gradients, rétropropagation et mise à jour avec Adam.
Chaque graine utilise quinze époques ; le checkpoint de NLL minimale sur la
sélection est retenu. Ni augmentation, ni poids préentraînés, ni dropout.

**Le CNN calibré partage exactement les mêmes poids.** Seule une température
positive est apprise sur la partition de calibration : `softmax(logits / T)`.
L'ordre des logits d'une image ne change pas. Les classes prédites, l'exactitude
et le macro-F1 restent donc identiques ; les probabilités peuvent changer.
Le classement des images par confiance peut toutefois changer en multiclasse.

### Ce que mesurent les scores

| Mesure | Interprétation |
|---|---|
| Exactitude, macro-F1 | Qualité des classes prédites ; plus haut est préférable |
| NLL | Pénalisation logarithmique de la probabilité attribuée à la vraie classe |
| Brier | Écart quadratique entre distribution prédite et étiquette ; somme sur les neuf classes, plage [0, 2] |
| ECE | Écart confiance–exactitude de la classe prédite, sur quinze intervalles de largeur égale |
| Risque-couverture | Taux d'erreur parmi les images conservées, selon la fraction acceptée |
| AURC | Moyenne des risques cumulés du classement ; plus bas est préférable |

La NLL et le Brier évaluent les distributions prédictives, pas uniquement la
calibration. L'ECE dépend du découpage en intervalles : les diagrammes montrent
aussi leurs effectifs. Une faible ECE n'implique pas une bonne classification.

### Erreurs et abstention

Les dix erreurs les plus confiantes sont sélectionnées automatiquement, par
confiance brute décroissante puis indice croissant en cas d'égalité. Les mêmes
images sont affichées avec leur confiance avant et après calibration. Les trois
graines sont conservées ; la graine 0 sert d'illustration principale, comme fixé
avant les expériences.

Pour l'abstention, le modèle accepte une image si sa confiance maximale atteint
le seuil. Le rapport distingue les seuils fixés avant les expériences et les
couvertures obtenues en classant le test. Aucun seuil exploré dans la démo ne
devient un seuil validé pour un usage réel. Si aucune image n'est acceptée,
l'exactitude sur les images acceptées est indéfinie.

## Reproduire

Environnement CPU : Python 3.12, Linux x86_64. Les versions utilisées sont figées
dans [requirements.txt](requirements.txt). Le téléchargement utilise directement
l'archive officielle MedMNIST ; sa somme MD5 est vérifiée.

```bash
git clone https://github.com/KenziBoughadou/pathmnist-confidence.git
cd pathmnist-confidence
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python experiments.py --smoke
python experiments.py --output-dir results/ma-reproduction
python report.py --results-dir results/ma-reproduction
```

L'essai court utilise 1 024 images d'entraînement, 256 de sélection et 256 de
calibration, deux époques CNN et dix itérations de régression logistique. Il
n'accède jamais au test officiel ; ses résultats ne mesurent pas la performance
scientifique des modèles.

Chaque tentative crée un nouveau dossier. Un chemin existant est refusé : les
historiques et les avertissements ne sont jamais remplacés par une relance.
L'environnement virtuel, l'archive complète des données et les checkpoints
restent locaux. Les sorties nécessaires au rapport et à la démo sont distribuées.

Pour lire l'étude enregistrée sans entraîner ni télécharger les données :

```bash
python report.py --results-dir results/study
python -m streamlit run app.py
```

Streamlit affiche une image du test, sa vraie classe, la prédiction, les neuf
probabilités avant/après calibration et la décision d'abstention au seuil choisi.
Les poids ne sont pas nécessaires pour explorer des prédictions déjà calculées.
Aucun téléchargement d'image personnelle, entraînement ou diagnostic n'est proposé.

## Lire le code

| Fichier | Rôle |
|---|---|
| [data.py](data.py) | Archive, séparation sélection/calibration, conversion des images |
| [models.py](models.py) | Architecture du petit CNN |
| [training.py](training.py) | Entraînement et extraction des logits |
| [metrics.py](metrics.py) | Température, métriques, classement et abstention |
| [experiments.py](experiments.py) | Ordre des expériences et sauvegardes |
| [report.py](report.py) | Recalcul des tableaux et figures depuis les sorties |
| [app.py](app.py) | Exploration des résultats enregistrés |
| [tests](tests/) | Vérifications des calculs et de la séparation des usages |

Les **20 tests validés** couvrent notamment les partitions, les métriques, la
reproductibilité sur données synthétiques, le rechargement des poids, l’ordre
sélection–calibration–test, les interruptions, le rapport et les contrôles Streamlit.
Le rapport se régénère depuis les archives sans entraînement ni inférence.

Les résultats conservent les indices des partitions, les versions, les empreintes
du code, les historiques, les températures et les logits. Les probabilités et
les indices d'origine permettent de recalculer les scores et de retrouver une
erreur sans charger le modèle.

## Données, limites et portée

PathMNIST utilise des patchs histologiques issus de NCT-CRC-HE-100K pour le
développement et de CRC-VAL-HE-7K pour le test. Le dépôt officiel décrit ce dernier
comme provenant d'un autre centre clinique. L'étude observe donc les probabilités
sur ce test distinct ; elle ne démontre pas une robustesse générale aux changements
de distribution et n'identifie pas la cause d'un éventuel écart entre partitions.

Les neuf codes utilisés dans les figures sont : ADI (tissu adipeux), BACK (fond),
DEB (débris), LYM (lymphocytes), MUC (mucus), MUS (muscle lisse), NORM (muqueuse
colique normale), STR (stroma associé au cancer), TUM (épithélium d'adénocarcinome
colorectal).

- Les observations sont des patchs, pas des patients indépendants. La division
  sélection/calibration ne garantit pas une séparation par patient, faute
  d'identifiants utilisés à cette fin.
- La résolution de 28 × 28 limite les informations disponibles. Aucun avis
  histopathologique indépendant n'interprète les erreurs.
- Trois graines décrivent la variabilité du CNN sur une partition fixe. Les
  répétitions sur les mêmes images ne constituent pas de nouveaux patients.
- La baseline linéaire et le CNN n'ont ni capacité ni budget de calcul identiques.
  La comparaison porte sur ces configurations ; aucun réglage optimal n'est revendiqué.
- La température est optimisée pour la NLL de calibration, pas pour l'ECE, le
  Brier ou le risque du test. Leur amélioration n'est pas garantie.
- Une bonne calibration globale ne garantit pas la fiabilité d'une prédiction
  individuelle ni une calibration correcte pour chaque classe.
- Les seuils de confiance ne constituent pas une garantie de risque. Les durées
  proviennent d'un CPU partagé et ne sont pas un benchmark matériel contrôlé.

Les images distribuées dans les résultats proviennent de PathMNIST sous CC BY 4.0.
Les sources, attributions et transformations figurent dans la
[notice](THIRD_PARTY_NOTICES.md).

## Références

- Yang et al. (2023), [MedMNIST v2](https://doi.org/10.1038/s41597-022-01721-8).
- Kather et al. (2019), [Predicting survival from colorectal cancer histology slides using deep learning: A retrospective multicenter study](https://doi.org/10.1371/journal.pmed.1002730).
- [Métadonnées officielles PathMNIST](https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py).
- Guo et al. (2017), [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html).

Le projet [Fashion-MNIST Study](https://github.com/KenziBoughadou/fashion-mnist-study)
reste une étude distincte de comparaison MLP/CNN. Il ne contient pas les expériences
de calibration présentées ici.
