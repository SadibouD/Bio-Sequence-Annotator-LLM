# src/metrics.py
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np

def calculate_sequence_metrics(y_true, y_pred, average='weighted'):
    """
    Calcule l'Accuracy, la Précision, le Rappel et le F1-Score pour un problème de classification.
    
    Args:
        y_true (list/array): Les vraies étiquettes.
        y_pred (list/array): Les étiquettes prédites par le modèle.
        average (str): Type de moyenne .
        
    Returns:
        dict: Un dictionnaire contenant les métriques calculées.
    """
    
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    return metrics

