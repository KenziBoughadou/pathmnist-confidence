# Protocole expérimental

## Question et périmètre

Un modèle plus performant produit-il aussi des probabilités mieux calibrées ?
Comment évolue le taux d'erreur lorsque les exemples les moins confiants sont
écartés ? L'étude compare exactement trois états : régression logistique,
CNN brut et le même CNN après temperature scaling. Aucun gain n'est présupposé.

Le protocole et ses paramètres dans [protocol.json](../protocol.json) sont fixés
avant les expériences complètes. Les essais courts servent à vérifier le code,
pas à choisir les architectures ou à optimiser leurs scores.

## Données et partitions

PathMNIST 28 × 28 comporte neuf classes et trois canaux couleur. L'archive
officielle contient 89 996 images d'entraînement, 10 004 de validation et 7 180
de test. Son MD5 est vérifié au téléchargement et avant utilisation.

| Partition | Usage exclusif |
|---|---|
| Entraînement officiel | Apprentissage des poids de la régression logistique et du CNN |
| 5 002 images de validation-sélection | Sélection du checkpoint CNN par NLL minimale |
| 5 002 images de calibration | Ajustement d'une seule température par CNN sélectionné |
| Test officiel | Une évaluation finale des trois états après toutes les sélections et calibrations |

La validation officielle est divisée en deux moitiés stratifiées avec la graine
2026. Les indices sont sauvegardés avant les entraînements. Les architectures
et hyperparamètres sont fixés ; aucune recherche supplémentaire n'est effectuée.
La baseline linéaire est également évaluée sur la sélection à titre descriptif,
sans changer ses réglages.

L'archive est téléchargée en entier mais les tableaux du test ne sont chargés
qu'après tous les entraînements et calibrations. Les figures et la démo utilisent
ensuite ces sorties enregistrées, sans nouvelle inférence ni ajustement.

## Modèles et prétraitement

Les pixels sont divisés par 255. Aucune statistique de normalisation n'est estimée,
aucune augmentation ni extraction de caractéristiques n'est appliquée.

- **Régression logistique multinomiale** : 2 352 pixels aplatis, solveur L-BFGS,
  régularisation L2 avec `C=1`, tolérance `1e-4`, au plus 300 itérations.
  Un seul entraînement est exécuté : répéter ce solveur déterministe avec trois
  étiquettes de graine ne mesurerait pas une variabilité d'initialisation comparable
  à celle du CNN. Un avertissement de non-convergence sera conservé et discuté.
- **CNN** : convolution 3→16, noyau 3 et padding 1, ReLU, max-pooling 2 ;
  convolution 16→32 identique, ReLU, max-pooling 2 ; aplatissement, dense 1568→64,
  ReLU, dense 64→9. Adam à `0.001`, autres paramètres par défaut, lots de 256,
  15 époques, graines 0, 1 et 2. Pas de dropout ni de weight decay.
- **CNN calibré** : poids inchangés, logits divisés par une température `T > 0`.
  Pour chaque graine, on minimise la NLL sur la calibration en optimisant `log(T)`
  dans `[-4, 4]`. Une solution proche d'une borne est signalée ; `T=1` est gardé
  si l'optimisation n'améliore pas la NLL de calibration.

Les trois CNN sont entraînés entièrement avant de calibrer leurs checkpoints.
Le checkpoint de plus faible NLL de sélection est retenu ; en cas d'égalité,
le premier est conservé. La boucle exécute toujours les quinze époques.

Les opérations déterministes sont activées, avec deux threads CPU et un générateur
indépendant pour le mélange des lots. Les versions et empreintes du code sont
archivées. L'identité numérique entre plateformes n'est pas garantie.

## Définitions des mesures

Pour les probabilités `p`, la vraie classe `y` et neuf classes :

- **Exactitude** : proportion de classes correctement prédites.
- **Macro-F1** : moyenne non pondérée des neuf F1 par classe ; zéro si le F1
  d'une classe est indéfini.
- **NLL** : moyenne de `-log(p[y])`, en logarithme naturel. Elle est calculée
  depuis les logits avec log-sum-exp pour éviter les problèmes numériques.
- **Brier multiclasse** : moyenne de `sum_k (p[k] - 1[y=k])²`. La somme sur
  les classes n'est pas divisée par neuf ; la plage est `[0, 2]`.
- **ECE** : calibration de la classe prédite avec 15 intervalles de confiance
  de largeur égale. Somme des écarts absolus entre exactitude et confiance
  moyennes de chaque intervalle, pondérés par ses effectifs. Intervalles fermés
  à gauche, ouverts à droite, sauf le dernier qui inclut 1. Les intervalles
  vides ont une contribution nulle et aucune exactitude artificielle.

La NLL et le Brier évaluent les distributions prédictives, pas uniquement leur
calibration. L'ECE dépend du découpage et ne décrit pas la calibration de toutes
les classes séparément. Les effectifs des intervalles accompagnent les diagrammes.

Les scores sont rapportés pour chaque graine du CNN, puis en moyenne et écart-type
d'échantillon (`ddof=1`). La régression logistique reste une mesure unique, sans
écart-type fabriqué. Les écarts entre CNN brut et calibré sont appariés par graine.
La même température positive conserve l'argmax, donc l'exactitude et le macro-F1
doivent rester identiques avant et après calibration.

## Abstention et erreurs

La confiance est la probabilité maximale. Les exemples sont classés par confiance
décroissante ; les indices officiels croissants départagent les ex aequo. Pour
les `k` premiers exemples, la couverture vaut `k/N` et le risque est leur taux
d'erreur. L'AURC est la moyenne des `N` risques cumulés, sans point artificiel à
couverture zéro. Une AURC plus faible est préférable.

Les tableaux utilisent les couvertures 100 %, 90 %, 75 % et 50 %, avec `ceil(cN)`
exemples. Ils évaluent un classement, pas un seuil opérationnel choisi en amont.
Des seuils fixes `0.5, 0.7, 0.8, 0.9, 0.95` sont aussi rapportés. Si aucun exemple
n'est accepté, exactitude et risque sont indéfinis, pas nuls. Ces analyses sont
descriptives ; elles ne garantissent aucun niveau de risque en déploiement.

Pour chaque CNN, les dix erreurs les plus confiantes selon ses probabilités
**brutes** sont montrées avec leur image, classe réelle, prédiction et confiances
avant/après calibration. Les mêmes indices sont utilisés pour cette comparaison.
Les erreurs classées par confiance calibrée sont également exportées. La graine
0 est choisie à l'avance pour la figure principale, sans masquer les autres graines.
Les paires classe réelle/prédite des erreurs de confiance ≥ 0,9 sont comptées
automatiquement pour les trois états.

La température peut changer le classement des confiances entre images en
multiclasse ; aucune amélioration de la courbe risque-couverture n'est présupposée.

## Coût, limites et conservation

La durée de la baseline inclut la conversion des pixels et l'ajustement. Celle
du CNN comprend ses époques, la sélection et les sauvegardes. L'optimisation de
la température est mesurée séparément. Le téléchargement et l'évaluation finale
sont exclus de ces durées. Le CPU est partagé : ces temps ne sont pas un benchmark
matériel contrôlé, et les budgets des deux modèles ne sont pas égaux.

Le test provient de CRC-VAL-HE-7K, décrit comme issu d'un autre centre clinique
que NCT-CRC-HE-100K utilisé pour le développement. Cela permet une observation
sur ce test distinct, sans démontrer une robustesse générale aux changements de
distribution. Les images sont des patchs, pas des patients indépendants ; aucun
identifiant patient n'est utilisé pour garantir une séparation par patient entre
sélection et calibration. Aucune validation clinique n'est réalisée.

Chaque tentative crée un nouveau dossier et conserve ses avertissements ou erreurs.
Une interruption bloque la suite du protocole ; aucune évaluation finale n'a lieu
si sélection et calibration n'ont pas abouti. Les tentatives incomplètes ne sont
pas comptées comme des expériences réussies. Aucune réécriture des résultats
antérieurs et aucune relance sélective d'une graine ne sont prévues.

## Références

- [Métadonnées et origine de PathMNIST](https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py).
- [Archive officielle](https://zenodo.org/records/10519652).
- Guo et al. (2017), [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html).
- [Régression logistique scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
