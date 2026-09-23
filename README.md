# TP1 — Détection de fake news

## Objectif
Comparer trois approches pour classer des déclarations en Fake ou Real :
- TF-IDF + régression logistique.
- DistilBERT gelé + régression logistique.
- DistilBERT fine-tuné.
- En bonus, évaluer un LLM (`gemini-3.5-flash-lite`) en zero-shot sur 100 textes.

## Fichiers
- tp1.ipynb : notebook contenant le code et les résultats.
- requirements.txt : bibliothèques et versions utilisées.
- README.md : instructions d'exécution.

Le compte-rendu est fourni séparément.

## Données
Dataset LIAR, chargé depuis Hugging Face :
rickpereira/liar

Les ensembles train, validation et test fournis sont conservés.

Binarisation :
- Fake (0) : pants-fire, false, barely-true.
- Real (1) : mostly-true, true.
- Les déclarations half-true sont exclues.

Après préparation :
- Train : 8126 textes.
- Validation : 1036 textes.
- Test : 1002 textes.

## Environnement utilisé
- Google Colab.
- Python 3.13.15.
- GPU NVIDIA Tesla T4 pour DistilBERT.
- CPU pour les régressions logistiques.
- Versions des bibliothèques dans requirements.txt.

Une connexion Internet est nécessaire pour télécharger les données
et le modèle préentraîné.

## Exécution sur Google Colab
1. Importer le notebook tp1.ipynb dans Colab.
2. Sélectionner un GPU dans les paramètres du type d'exécution,
   si disponible.
3. Importer requirements.txt dans les fichiers de la session.
4. Installer les dépendances dans une cellule :

   %pip install -r requirements.txt

5. Redémarrer la session si nécessaire après l'installation.
6. Exécuter les cellules du notebook dans l'ordre, du début à la fin.

Le code peut utiliser le CPU si aucun GPU n'est disponible,
mais les étapes DistilBERT seront plus lentes.

Les versions enregistrées correspondent à l'environnement Colab utilisé.
La version de PyTorch avec le suffixe +cu128 peut nécessiter
une installation adaptée sur un autre environnement.
## Bonus — LLM en zero-shot

Le modèle `gemini-3.5-flash-lite` est utilisé via l’API Google en **zero-shot** : le prompt demande de classer chaque déclaration en Fake ou Real, sans fournir d’exemples et sans entraîner le modèle.

### Configuration de la clé API
1. Créer une clé API dans Google AI Studio, rubrique **API Keys**.
2. Dans Colab, ouvrir **Secrets** (icône de clé).
3. Ajouter la clé avec le nom exact attendu par le code du notebook, puis autoriser l’accès au notebook.
4. Exécuter les cellules de la partie LLM avec une connexion Internet.

La clé est lue depuis les Secrets et ne doit pas être écrite directement dans le code ni publiée dans le dépôt.

### Évaluation
Le LLM est évalué sur **100 textes**, avec une **accuracy de 0.75** et un **F1-macro de 0.7457**. Ces résultats ne sont pas directement comparables à ceux des trois autres approches, évaluées sur **1002 textes**. Aucun entraînement local ni GPU Colab n’est nécessaire pour les appels API ; le traitement est effectué à distance.
## Reproductibilité
La seed est fixée à 42 pour Python, NumPy et PyTorch.
Les régressions logistiques utilisent random_state=42.
Le Trainer utilise seed=42.

Des différences peuvent subsister selon le matériel et les bibliothèques.
Les temps d'exécution dépendent des ressources disponibles.

## Évaluation
Les modèles sont comparés avec :
- Accuracy.
- F1-macro.
- Matrices de confusion.
- Temps d'entraînement.
- Temps d'extraction des embeddings pour DistilBERT gelé.
- Temps moyen de prédiction par texte sur un lot de 100 textes.

Le temps d'entraînement de la baseline exclut la transformation TF-IDF.
Le temps d'extraction des embeddings inclut train, validation et test.
Le temps de fine-tuning inclut les évaluations et sauvegardes du Trainer.

Le coût indiqué de 0 euro correspond à une exécution sans paiement
sur les ressources gratuites utilisées ; il ne signifie pas
une absence d'utilisation de GPU.

## Résultats enregistrés
| Approche                                | Accuracy | F1-macro |
|-----------------------------------------|----------|----------|                             
| TF-IDF + régression logistique          | 0.6527   | 0.6395   |
| DistilBERT gelé + régression logistique | 0.6537   | 0.6487   |
| DistilBERT fine-tuné                    | 0.6447   | 0.6441	 |

## Limites
Les déclarations sont courtes et leur vérification peut nécessiter
des preuves externes. Les modèles utilisent uniquement leur texte.
Le fine-tuning présente un sur-apprentissage, surveillé à l'aide
des losses et limité par un arrêt anticipé.
