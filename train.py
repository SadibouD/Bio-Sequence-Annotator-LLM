# train.py
import torch
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
import numpy as np
from src.metrics import calculate_sequence_metrics

#1. Configuration et Constantes

#MODEL_NAME = "microsoft/biogpt"
MODEL_NAME = "prajjwal1/bert-tiny"
DATA_PATH = "./data/human_dna_dataset.csv"  
#MODEL_OUTPUT_DIR = "./models/biogpt_finetuned"
MODEL_OUTPUT_DIR = "./models/bert_tiny_finetuned"
NUM_CLASSES = 7  #Le jeu de données a 7 classes (0 à 6)

#2. Chargement et Préparation des Données

def load_and_prepare_dataset(data_path, tokenizer):
    """Charge le CSV, le convertit en Dataset Hugging Face et le tokenise."""
    try:
    
        df = pd.read_csv(data_path, sep=',')

        print(df.head())  # Affiche les premières lignes pour vérifier le chargement
       
        df = df[['sequence', 'label']]
    except FileNotFoundError:
        print(f"ERREUR: Fichier de données introuvable à {data_path}")
        
        return None, None
    except KeyError:
        print("ERREUR: Le CSV doit avoir les colonnes 'sequence' et 'label'.")
        return None, None

    #Séparer en train/validation
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'])

    # Convertir en Dataset Hugging Face
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # Fonction de tokenisation
    MAX_SEQ_LENGTH=512
    def tokenize_function(examples):
        return tokenizer(
        examples['sequence'], 
        padding="max_length", 
        truncation=True, 
        max_length=MAX_SEQ_LENGTH
    )

    print("Tokenisation des données...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    
    return tokenized_train, tokenized_val

def compute_metrics(eval_pred):
    """
    Fonction appelée par le Trainer pour calculer les métriques lors de l'évaluation.
    """
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metrics = calculate_sequence_metrics(labels, predictions, average='weighted')      # 'weighted' est utilisé pour les classes déséquilibrées
    
    return {
        'accuracy': metrics['accuracy'],
        'f1': metrics['f1_score'],
        'precision': metrics['precision'],
        'recall': metrics['recall']
    }

#3. Script Principal d'Entraînement

if __name__ == "__main__":
    print(" Démarrage du Fine-Tuning du Modèle Bio-GPT :")

    # Charger le Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Charger les données
    train_dataset, val_dataset = load_and_prepare_dataset(DATA_PATH, tokenizer)

    if train_dataset is not None:
        # Charger le Modèle pour la Classification
        print(f"Chargement de {MODEL_NAME} en mode Classification (7 classes)...")
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, 
            num_labels=NUM_CLASSES
        )

       
        training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        num_train_epochs=3,  #1
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_strategy="epoch",        
        eval_steps=500,                
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",  
        greater_is_better=True,
        )
        
        # Initialiser le Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )

        # Lancer le Fine-Tuning !
        print("Lancement du Fine-Tuning :")
        trainer.train()

        # Sauvegarder le modèle final
        print(f"Entraînement terminé. Modèle sauvegardé dans {MODEL_OUTPUT_DIR}")
        trainer.save_model(MODEL_OUTPUT_DIR)
        tokenizer.save_pretrained(MODEL_OUTPUT_DIR)