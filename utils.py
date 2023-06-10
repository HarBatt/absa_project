from sklearn.metrics import f1_score

def simple_accuracy(preds, labels):
    """
    Calculates accuracy of predictions.
    
    Args:
        preds: Model predictions.
        labels: Ground truth labels.
        
    Returns:
        The accuracy of the predictions.
    """
    return (preds == labels).mean()


def accuracy_and_f1_score(preds, labels):
    """
    Calculates accuracy and F1-score of predictions.
    
    Args:
        preds: Model predictions.
        labels: Ground truth labels.
        
    Returns:
        A dictionary containing the accuracy and F1-score of the predictions.
    """
    accuracy = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "accuracy": accuracy,
        "f1_score": f1
    }


def calculate_metrics(preds, labels):
    """
    Computes metrics of predictions.
    
    Args:
        preds: Model predictions.
        labels: Ground truth labels.
        
    Returns:
        The calculated metrics (accuracy and F1-score).
    """
    return accuracy_and_f1_score(preds, labels)