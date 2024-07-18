import re 

def calc_metrics(ground_truth, predictions):

    set_ground_truth = set(ground_truth)
    set_predictions = set(predictions)

    true_positives = set_ground_truth.intersection(set_predictions)
    false_positives = set_predictions.difference(set_ground_truth)
    false_negatives = set_ground_truth.difference(set_predictions)

    # Cálculo de métricas
    precision = len(true_positives) / (len(true_positives) + len(false_positives))
    recall = len(true_positives) / (len(true_positives) + len(false_negatives))
    f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1




def evaluate(masked, generated):
    """ 
    Input: 
        - masked (str): Ground_truth text
        - generated(str): Text to be evaluated

    Output:
        - Precision, Recall and F1 (float)
    """
    ground_truth = re.findall(r'\[\*\*(.*?)\*\*\]', masked)
    predictions = re.findall(r'\[\*\*(.*?)\*\*\]', generated)
    predictions = predictions[0:3]+predictions[5:]

    return calc_metrics(ground_truth, predictions)

