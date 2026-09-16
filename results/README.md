# Tentatives expérimentales

Les dossiers sont conservés séparément. Une relance ne remplace pas les
historiques ni les avertissements de la tentative précédente.

| Dossier | Nature | État |
|---|---|---|
| [reference](reference/) | Première tentative complète | Interrompue par SIGTERM pendant le premier CNN ; aucune évaluation du test |
| [study](study/) | Relance intégrale, même protocole | Terminée : une régression logistique, trois CNN, trois températures ; [rapport complet](study/report/README.md) |

La première tentative a débuté le 16 septembre 2026 à 11:52:42 UTC et a reçu le
signal d'arrêt à 11:59:01 UTC. La cause externe du signal n'est pas établie.
La baseline linéaire avait atteint sa limite de 300 itérations avec un avertissement
de non-convergence ; deux époques CNN étaient enregistrées. Le signal est survenu
pendant une rétropropagation, comme indiqué dans le [journal](reference/error.txt).

La relance reprend tous les entraînements, avec les mêmes paramètres, partitions
et graines, dans un processus détaché de la session de commande. Elle ne reprend
pas un checkpoint partiel et ne sélectionne aucune graine sur la base de son score.
Les essais courts de développement sont séparés dans un dossier local ignoré ;
ils n'utilisent pas le test officiel.
