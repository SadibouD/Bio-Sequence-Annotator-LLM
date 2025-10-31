# Classification de Séquences ADN par Affinage de Modèles Transformers (BERT/Bio-GPT)

Ce projet démontre la chaîne de traitement complète (pipeline MLOps) pour l'affinage (*fine-tuning*) et l'évaluation rigoureuse de modèles de langue (LLM) pour une tâche de classification bio-informatique.

L'objectif est d'entraîner un modèle Transformer à classifier des séquences d'ADN humain en 7 familles de gènes distinctes, en utilisant un jeu de données public de Kaggle.

---

##  Objectifs et Compétences Démontrées

* **Ingénierie des Données :** Chargement et nettoyage d'un jeu de données complexe (`.csv` mal formaté) avec Pandas.
* **Pipeline de Fine-Tuning :** Utilisation de l'écosystème Hugging Face (`Trainer`, `Datasets`, `Tokenizers`) pour entraîner un modèle de classification de séquences.
* **Gestion des Modèles :** Capacité à gérer les contraintes de différents modèles (de `prajjwal1/bert-tiny` en local à `microsoft/biogpt` sur GPU distant).
* **Évaluation Rigoureuse :** Calcul des métriques de performance réelles (**Accuracy**, **F1-Score**, Précision, Rappel) sur un jeu de test avec `scikit-learn`.

---

## Structure du Projet

* `train.py`: Script principal pour le fine-tuning. Il charge les données, tokenise, entraîne le modèle et sauvegarde le classificateur final dans `/models`.
* `main.py`: Script principal pour l'évaluation. Il charge le modèle fine-tuné et l'évalue sur le jeu de test pour générer les métriques de performance.
* `src/llm_helper.py`: Gère le chargement du modèle fine-tuné et la logique d'inférence (classification).
* `src/metrics.py`: Outil de calcul des métriques de performance (basé sur `scikit-learn`).
* `data/human_dna_dataset.csv`: Le jeu de données de benchmark (Kaggle).
* `requirements.txt`: Les dépendances Python.
* `models/`: (Ignoré par `.gitignore`) Dossier de sortie pour les modèles entraînés.

---

## Résultats et Analyse

L'entraînement a été effectué sur deux modèles pour comparer leur capacité à gérer cette tâche complexe :

### 1. Modèle `prajjwal1/bert-tiny` (Local, 3 Époques)

Ce modèle très léger (4.4M de paramètres) a été entraîné localement.

* **Accuracy:** 30.71%
* **F1-Score:** 0.1443

**Analyse :** Le modèle a subi un **effondrement de classe** (il prédit toujours la classe la plus fréquente, la '6'). Son F1-Score (14.4%) est équivalent à une prédiction aléatoire (1/7 = 14.3%). **Conclusion : Ce modèle est trop petit pour cette tâche.**

### 2. Modèle `microsoft/biogpt` (Colab GPU, 1 Époque)

Ce modèle lourd (1.5G de paramètres), spécialisé en biologie, a été entraîné sur Google Colab.

* **Accuracy:** 33.10%
* **F1-Score:** 0.2122

**Analyse :** Avec **une seule époque** d'entraînement, Bio-GPT surpasse largement le score aléatoire (14.3%). Cela prouve que le modèle **apprend activement** et que le pipeline est correct. Des époques supplémentaires amélioreraient considérablement ce score ( à tester !).

---

## Installation et Utilisation

### 1. Installation

Clonez le dépôt et installez les dépendances :
```bash
git clone [https://github.com/SadibouD/Bio-Sequence-Annotator-LLM.git](https://github.com/SadibouD/Bio-Sequence-Annotator-LLM.git)
cd Bio-Sequence-Annotator-LLM
pip install -r requirements.txt 