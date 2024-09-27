import json
import pandas as pd

class MetricsDict:
    def __init__(self, name_model = "Unassigned"):
        self._list_metrics = []
        self._name_model = name_model
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

    def set_filename(self, filename):
        self._metrics_data["filename"] = filename
    
    def store_metrics(self, real, predicted,fails, miss):
        self._list_metrics.append(self._metrics_data)
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
        keys_not_present_in_dict2 = list(set(dict1.keys()) - set(dict2.keys()))
        
        disagreement_in_values = {}
        for key in dict1.keys() & dict2.keys():
            if dict1[key] != dict2[key]:
                disagreement_in_values[key] = (dict1[key], dict2[key])

        return disagreement_in_values, keys_not_present_in_dict2

    def get_array_ground_truth(self, filename):
        df = pd.read_csv("data/carmen/tsv/replaced/CARMEN-I_replaced_anon.tsv", sep='\t')
        filtered_df = df[df['name'] == filename]
        result = [{row['text']: row['tag']} for _, row in filtered_df.iterrows()]
        json_result = json.dumps(result, ensure_ascii=False, indent=4)
        print(json_result)
        return json_result

    def calculate_classification_metrics(self,  filename, array_generated):
        print("\n classification metrics \n")
        print(array_generated)
        array_ground_truth = self.get_array_ground_truth(filename.split(".")[0])
        merged_dict = {}
        for d in array_generated:
            merged_dict.update(d)
        diferencias, unicas_dict1 = self.compute_classification_agreement(array_ground_truth, array_generated)
        self.store_metrics(array_ground_truth, merged_dict, diferencias, unicas_dict1)
    
    def save_metrics(self):
        df = pd.DataFrame(self._list_metrics)
        df.to_csv(f"data/metrics/{self._name_model}_metrics.csv")