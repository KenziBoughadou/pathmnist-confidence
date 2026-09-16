# Attribution des données et images

Les images contenues dans `results/study/evaluation_images.npz` et dans les
figures proviennent de **PathMNIST**, sous-ensemble de **MedMNIST v2**, distribué
sous [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

- Fournisseurs de MedMNIST : Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu,
  Lin Zhao, Bilian Ke, Hanspeter Pfister et Bingbing Ni.
- Référence : *MedMNIST v2 — A large-scale lightweight benchmark for 2D and 3D
  biomedical image classification*, Scientific Data 10, 41 (2023),
  [DOI](https://doi.org/10.1038/s41597-022-01721-8).
- Distribution utilisée : [archive officielle MedMNIST](https://zenodo.org/records/10519652),
  fichier `pathmnist.npz`, version 28 × 28.
- Origine des patchs : Jakob Nikolas Kather et collaborateurs, NCT-CRC-HE-100K
  et CRC-VAL-HE-7K ; [données sources](https://zenodo.org/records/1214456) et
  [publication PLOS Medicine (2019)](https://doi.org/10.1371/journal.pmed.1002730).
- [Métadonnées PathMNIST et indication de licence](https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py).

MedMNIST a redimensionné les images sources vers 28 × 28 pixels. Ce dépôt
redistribue les images du test à cette résolution, sans retouche du contenu.
Les figures et l'application les agrandissent pour affichage et ajoutent des
étiquettes et des mesures issues de l'expérience. Les traductions françaises
des classes sont propres à ce dépôt.

La redistribution n'implique aucune approbation du projet par les auteurs des
données. Les résultats sont académiques et ne constituent pas un outil clinique.
