import spacy
import warnings
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore", message="\[W008\] Evaluating Doc.similarity based on empty vectors")

class Metrics:
    def __init__(self, name_model = "Unassigned"):
        self._nlp = spacy.load("es_core_news_md")
        self._name_model = name_model
        self._list_metrics = []
        self._metrics_data = {
            "filename":             None,
            "precision":            None,
            "recall":               None,
            "f1":                   None,
            "cosine_sim":           None,
            "levenshtein_distance": None,
            "labels":               None,
            "inv_levenshtein":      None,
            "overall":              None,
        }

    def set_filename(self, filename):
        self._metrics_data["filename"] = filename

    def embedding_similarity(self, str1, str2, threshold=0.8):
        doc1 = self._nlp(str1)
        doc2 = self._nlp(str2)
        similarity = doc1.similarity(doc2)
        return similarity >= threshold
    
    def erase_adverbs_determinants(self, texto):
        doc = self._nlp(texto)
        tokens_filtrados = [token.text for token in doc if token.pos_ not in ('ADP', 'DET')]
        return ' '.join(tokens_filtrados)
    
    def levenshtein_distance(self, s1, s2, show_progress=True):
        # Usar tqdm solo si show_progress es True
        iterable = tqdm(s1) if show_progress else s1

        if len(s1) < len(s2):
            s1, s2 = s2, s1
        if len(s2) == 0:
            self._metrics_data["levenshtein_distance"]  = len(s1)
            return self._metrics_data["levenshtein_distance"]

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(iterable):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        self._metrics_data["levenshtein_distance"] = previous_row[-1]

    def get_cos_sim(self, text_hoped, text_generated):
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w[\w\-/]*\b")
        tfidf_matrix = vectorizer.fit_transform([text_hoped, text_generated])
        try:
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            self._metrics_data["cosine_sim"] = cosine_sim[0][0]
        except: 
            self._metrics_data["cosine_sim"] = 0.0
        return self._metrics_data["cosine_sim"]

    def get_precison(self, true_positives, false_positives):
        self._metrics_data["precision"] = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

    def get_recall(self, true_positives, false_negatives):
        self._metrics_data["recall"] = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    def get_f1(self, precision, recall):
        self._metrics_data["f1"] = 2 * (precision * recall) / (precision + recall)

    def get_classification_metrics(self, ground_truth, predictions):
        # Convertir arrays de ground_truth y predictions a listas de str
        ground_truth_processed = np.array([self.erase_adverbs_determinants(str(item)) for item in ground_truth])
        predictions_processed = np.array([self.erase_adverbs_determinants(str(item)) for item in predictions])
    
        # Crear matrices de similitud de coseno y embedding 
        get_cos_sim_vectorized = np.vectorize(lambda gt, pred: self.get_cos_sim(str(gt), str(pred)))
        embedding_similarity_vectorized = np.vectorize(lambda gt, pred: self.embedding_similarity(str(gt), str(pred)))
    
        cosine_results = get_cos_sim_vectorized(ground_truth_processed[:, None], predictions_processed[None, :])
        embedding_results = embedding_similarity_vectorized(ground_truth_processed[:, None], predictions_processed[None, :])
        
        # Promediar las similitudes
        avg_similarities = (cosine_results + embedding_results) / 2
    
        # Determinar verdaderos positivos
        matches = avg_similarities > 0.5
        true_positives = np.sum(np.any(matches, axis=1))
    
        # Determinar falsos negativos
        false_negatives = len(ground_truth) - true_positives
    
        # Determinar falsos positivos
        predicted_matches = np.any(matches, axis=0)
        false_positives = len(predictions) - np.sum(predicted_matches)
        
        # Determinar verdaderos negativos
        true_negatives = None
        return true_positives, true_negatives, false_positives, false_negatives

    def calc_metrics(self, ground_truth, predictions):
        (
            true_positives,
            _, 
            false_positives, 
            false_negatives 
        ) = self.get_classification_metrics(ground_truth, predictions)

        self.get_precison(true_positives, false_positives)
        self.get_recall(true_positives, false_negatives)
        self.get_f1(self._metrics_data["precision"], self._metrics_data["recall"])

    def evaluate(self, masked, generated):
        ground_truth = re.findall(r'\[\*\*(.*?)\*\*\]', masked)
        predictions = re.findall(r'\[\*\*(.*?)\*\*\]', generated)
        self._metrics_data["labels"] = [ground_truth, predictions]
        self.calc_metrics(ground_truth, predictions) # Calculates precision, recall, f1

    def calculate_inv_levenshtein(self):
        if int(self._metrics_data["levenshtein_distance"]) == 0:
            self._metrics_data["inv_levenshtein"] = 1
        else: 
            self._metrics_data["inv_levenshtein"] = (1/self._metrics_data["levenshtein_distance"])
    
    def calculate_overall(self):
        self._metrics_data["overall"] = self._metrics_data["precision"] + self._metrics_data["recall"] +  self._metrics_data["f1"] + self._metrics_data["cosine_sim"] + self._metrics_data["inv_levenshtein"]

    def calculate(self, ground_truth, generated):
        self.evaluate(ground_truth, generated)
        self.get_cos_sim(ground_truth, generated)
        self.levenshtein_distance(ground_truth, generated)
        self.calculate_inv_levenshtein()
        self.calculate_overall()

    def store_metrics(self):
        self._list_metrics.append(self._metrics_data)
        self._metrics_data = {
            "filename":             None,
            "precision":            None,
            "recall":               None,
            "f1":                   None,
            "cosine_sim":           None,
            "levenshtein_distance": None,
            "labels":               None,
            "inv_levenshtein":      None,
            "overall":              None,
        }

    def get_metrics(self):
        return self._metrics_data

    def save_metrics(self):
        df = pd.DataFrame(self._list_metrics)
        df.to_csv(f"data/metrics/{self._name_model}_metrics.csv")