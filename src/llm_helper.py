# src/llm_helper.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax

#  1. CONFIGURATION 
# Chemin vers le modèleentraîné avec train.py
MODEL_PATH = "./models/biogpt_finetuned" 
#MODEL_PATH = "./models/bert_tiny_finetuned"  #changement pour bert-tiny
MAX_SEQ_LENGTH = 512 

print(f"Chargement du modèle Fine-Tuné : {MODEL_PATH}")

try:
    #2. CHARGEMENT DU MODÈLE FINE-TUNÉ 
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() 
    print(f"Modèle fine-tuné chargé sur: {device}")

except OSError:
    print(f"ERREUR: Modèle fine-tuné introuvable à {MODEL_PATH}")
    print("Veuillez d'abord lancer 'python train.py' pour entraîner et sauvegarder le modèle.")
    raise
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    raise

#3. FONCTION DE PRÉDICTION (Classification)

def classify_sequence(sequence_text):
    """
    Prédit la classe (0-6) pour une séquence ADN donnée
    en utilisant le modèle fine-tuné.
    """
    
    #Tokenisation de la séquence d'entrée
    inputs = tokenizer(
        sequence_text, 
        return_tensors="pt",
        padding="max_length", 
        truncation=True, 
        max_length=MAX_SEQ_LENGTH
    )
    
    # Envoie des entrées sur  (CPU/GPU)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Inférence
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 'outputs.logits' contient les scores bruts pour les 7 classes
    logits = outputs.logits
    
    # Obtenir la prédiction de la classe avec le score le plus élevé
    prediction = torch.argmax(logits, dim=-1).item()
    
    # probabilities = softmax(logits, dim=-1)
    
    return prediction