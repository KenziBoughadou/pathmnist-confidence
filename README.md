# PathMNIST Confidence

**Quand un classifieur d'images histologiques se dit sûr de lui, a-t-il raison ?**

Sur PathMNIST (neuf types de tissus colorectaux), je compare une régression
logistique et un petit CNN. J'étudie ensuite la fiabilité de leurs probabilités :
calibration par température, puis abstention, c'est-à-dire refuser les images
les moins confiantes.

Projet académique personnel · Python, PyTorch, scikit-learn, Streamlit ·
**aucun usage clinique**

[Rapport complet](results/study/report/README.md) · [Protocole](docs/PROTOCOL.md)

## En bref

- **Le CNN classe nettement mieux** : 77,5 % d'exactitude (± 2,1 sur trois
  graines) contre 52,8 % pour la régression logistique.
- **Ses probabilités sont déjà presque calibrées en moyenne.** La température
  apprise reste entre 0,93 et 0,96, et la calibration ne change presque rien :
  la NLL du test augmente même légèrement, de 0,01 à 0,03 selon la graine.
- **Le vrai problème, ce sont les erreurs très confiantes.** Selon la graine,
  4,5 à 9,7 % des prédictions données avec au moins 90 % de confiance sont
  fausses. Elles se concentrent sur quelques confusions (débris → muscle lisse,
  mucus → fond, muscle lisse → tumeur), que la calibration ne corrige pas.

![Dix erreurs les plus confiantes du CNN, graine 0](results/study/report/cnn-seed-0-top-errors.png)

*Les dix erreurs les plus confiantes du CNN (graine 0), sélectionnées
automatiquement : toutes sont prédites avec plus de 99,98 % de confiance.
La dernière image, à la coloration atypique, rappelle que les colorations
varient d'une lame à l'autre.*

## Ce que j'en retiens

- **Une bonne calibration moyenne ne rend pas chaque prédiction fiable.**
  L'ECE du CNN est faible (0,036), alors que des erreurs à 99,99 % de confiance
  subsistent.
- **Le temperature scaling corrige un biais global de confiance.** Quand ce biais
  est déjà faible, il n'a presque rien à corriger. Il ne touche pas non plus
  aux confusions entre classes qui produisent les erreurs confiantes.
- **Un seul modèle ne suffit pas pour juger sa confiance.** Le nombre d'erreurs
  à au moins 90 % de confiance varie presque du simple au triple entre graines
  (120 à 334), à exactitude proche. C'est un argument pour les ensembles de modèles.

## Résultats

Test officiel de 7 180 images, évalué une seule fois après tous les entraînements
et calibrations. Pour le CNN : moyenne ± écart-type sur trois graines.

| Modèle | Exactitude (%) | Macro-F1 | NLL ↓ | Brier ↓ | ECE ↓ |
|---|---:|---:|---:|---:|---:|
| Régression logistique | 52,84 | 0,435 | 1,244 | 0,577 | 0,035 |
| CNN | 77,47 ± 2,12 | 0,708 ± 0,019 | 0,800 ± 0,095 | 0,334 ± 0,030 | 0,036 ± 0,012 |
| CNN + température | 77,47 ± 2,12 | 0,708 ± 0,019 | 0,816 ± 0,104 | 0,335 ± 0,031 | 0,038 ± 0,019 |

La température ne change pas la classe prédite : exactitude et macro-F1 restent
identiques, seules les probabilités changent.

**Abstention.** En écartant la moitié des images les moins confiantes, le taux
d'erreur passe de 21-25 % à 6-11 % selon la graine. La régression logistique,
pourtant bien moins exacte, garde une ECE comparable : l'ECE seule ne permet
pas de comparer deux modèles.

![Risque selon la couverture, graine 0](results/study/report/risk-coverage-seed-0.png)

*Taux d'erreur parmi les images acceptées selon la fraction acceptée (graine 0).
Pour cette graine, les ~20 % d'images les plus confiantes contiennent plus
d'erreurs que les 50 % les plus confiantes. Les graines 1 et 2 ne présentent
pas cette inversion (voir le [rapport](results/study/report/README.md)).*

## Méthode

```text
Entraînement officiel (89 996) ─── apprentissage des poids

Validation officielle ┬─────────── choix de l'époque du CNN : 5 002 images
                      └─────────── apprentissage de la température : 5 002 images

Test officiel (7 180) ──────────── évaluation finale, lue une seule fois
```

- **Régression logistique** sur les 2 352 pixels (L2, L-BFGS, 300 itérations).
- **CNN** : deux blocs convolution–ReLU–pooling puis deux couches denses,
  106 089 paramètres, Adam pendant 15 époques, trois graines. L'époque retenue
  minimise la NLL sur la partition de sélection.
- **Température** : une seule valeur par CNN, apprise en minimisant la NLL sur
  la partition de calibration, distincte de celle de sélection.
- **Mesures** : exactitude, macro-F1, NLL, score de Brier, ECE sur 15 intervalles,
  courbes risque-couverture.

Le protocole a été fixé avant les expériences et une tentative interrompue est
[conservée](results/README.md). Le [protocole](docs/PROTOCOL.md) détaille les
définitions et règles de décision. Une application Streamlit ([app.py](app.py))
permet d'explorer les prédictions enregistrées image par image.

## Limites

- **Le CNN est sous-entraîné.** À la 15e époque, il n'atteint qu'environ 80 %
  d'exactitude sur l'entraînement et sa perte baisse encore. À titre de repère,
  un ResNet-18 atteint environ 90 % sur ce test dans MedMNIST v2. Les conclusions
  décrivent ce modèle, pas un CNN bien réglé.
- **La régression logistique n'a pas convergé** en 300 itérations : ce n'est
  pas une baseline linéaire optimisée.
- **Trois graines sur une seule partition** : les ± sont des écarts-types,
  pas des intervalles de confiance.
- **Des patchs 28 × 28, pas des patients**, sans relecture des erreurs par
  un pathologiste.
- **Pas de validation multicentrique.** MedMNIST décrit le test comme venant d'un
  autre centre, mais l'[archive d'origine](https://zenodo.org/records/1214456)
  indique le même centre (NCT) avec des patients différents.

## Pistes pour une v2

- Entraîner correctement le CNN (normalisation, augmentation des couleurs,
  scheduler, plus d'époques) et le comparer à des modèles préentraînés,
  dont un modèle de fondation en histopathologie.
- Combiner les trois graines en ensemble, ce qui vise directement l'instabilité
  des erreurs confiantes.
- Remplacer les seuils de confiance par de la prédiction conforme, qui donne
  une garantie de couverture.

Images PathMNIST sous licence CC BY 4.0 ; sources et attributions dans la
[notice](THIRD_PARTY_NOTICES.md).
