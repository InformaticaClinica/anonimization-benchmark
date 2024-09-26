

class MetricsDict:
    def __init__(self, name_model = "Unassigned"):
        self._metrics_data = {
            "filename":         None,
            "total":            None, 
            "correct":          None, 
            "total_fails":      None, 
            "precision":        None, 
            "real":             None, 
            "predicted":        None, 
            "fails":            None, 
            "miss":             None            
        }

    
    def store_metrics(self, real, predicted,fails, miss):
        self._metrics_data["total"] = len(real.keys())
        total_fails = len(fails.keys()) + len(miss)
        self._metrics_data["correct"] = len(real.keys()) - total_fails
        self._metrics_data["total_fails"] = total_fails
        self._metrics_data["precision"] = 100 - (total_fails * 100 / len(real.keys()))
        self._metrics_data["real"] = real
        self._metrics_data["predicted"] = predicted
        self._metrics_data["fails"] = fails
        self._metrics_data["miss"] = miss
    
    def compute_classification_agreement(self, dict1, dict2):
        # Look for key disagreement
        keys_not_present_in_dict2 = list(set(dict1.keys()) - set(dict2.keys()))
        
        # look for misclassifications
        disagreement_in_values = {}
        for key in dict1.keys() & dict2.keys():
            if dict1[key] != dict2[key]:
                disagreement_in_values[key] = (dict1[key], dict2[key])

        return disagreement_in_values, keys_not_present_in_dict2

    def calculate_classification_metrics(self, array_ground_truth, array_generated):
        merged_dict = {}
        for d in array_generated:
            merged_dict.update(d)
        diferencias, unicas_dict1 = self.compute_classification_agreement(array_ground_truth, merged_dict)
        self.store_metrics(array_ground_truth, merged_dict, diferencias, unicas_dict1)