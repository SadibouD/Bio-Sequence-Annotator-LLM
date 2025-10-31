# main.py
import pandas as pd
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from src.llm_helper import classify_sequence  
from src.metrics import calculate_sequence_metrics
from tqdm import tqdm

# 1. Configuration 
DATA_PATH = "./data/human_dna_dataset.csv"
#MODEL_PATH = "./models/biogpt_finetuned" 
MODEL_PATH = "./models/bert_tiny_finetuned"

# 2. Chargement et Préparation des Données de Test
def load_test_data(data_path):
    """Charge le jeu de données de test (CSV propre)."""
    try:
        df = pd.read_csv(data_path, delim_whitespace=True) 
        if 'sequence' not in df.columns:
             df = pd.read_csv(data_path) 

    except FileNotFoundError:
        print(f"ERREUR: Fichier de données introuvable à {data_path}")
        return None
    except Exception as e:
        print(f"Erreur lors de la lecture du CSV : {e}")
        return None

    if 'class' in df.columns:
        df = df.rename(columns={'class': 'label'})

    if 'sequence' not in df.columns or 'label' not in df.columns:
        print(f"ERREUR: Colonnes 'sequence' ou 'label' introuvables après lecture. Colonnes: {df.columns.tolist()}")
        return None

    # Séparer en train/validation
    _, test_df = train_test_split(df, test_size=0.2, stratify=df['label'])

    print(f"Jeu de données de test chargé : {len(test_df)} échantillons.")
    return test_df

#3. Script Principal d'Évaluation

if __name__ == "__main__":
    print(f"Démarrage de l'Évaluation du Modèle Fine-Tuné :")
    
    test_data = load_test_data(DATA_PATH)
    
    if test_data is not None:
        true_labels = []
        predicted_labels = []

        print("Lancement des prédictions sur le jeu de test...")

        for index, row in tqdm(test_data.iterrows(), total=len(test_data)):
            sequence = row['sequence']
            true_label = row['label']
            
            
            try:
                predicted_label = classify_sequence(sequence)
            except Exception as e:
                print(f"Erreur lors de la classification de la séquence {index}: {e}")
                predicted_label = -1 
            
            true_labels.append(true_label)
            predicted_labels.append(predicted_label)

        
        print("\n" + "="*50)
        print("        ÉVALUATION FINALE DU MODÈLE")
        print("="*50)
        
        metrics = calculate_sequence_metrics(true_labels, predicted_labels)
        
        print(f"{'Vraies Étiquettes (10 premières)':<30}: {true_labels[:10]}")
        print(f"{'Prédictions (10 premières)':<30}: {predicted_labels[:10]}")
        
        print("\n--- RÉSULTATS CHIFFRÉS (PERFORMANCE RÉELLE) ---")
        for key, value in metrics.items():
            print(f"{key.upper():<12}: {value:.4f}")
            
       