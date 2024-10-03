import json
import pandas as pd
import os 
import numpy as np

class MetricsDict:
    def __init__(self, name_model = "Unassigned"):
        self._list_metrics = []
        self._name_model = name_model
        self._metrics_data = {
            "filename":         None,
            "total":            None, 
            "total_correct":    None, 
            "total_fails":      None, 
            "precision":        None, 
            "ground_truth":     None, 
            "generated":        None, 
            "added":            None, 
            "missing":          None, 
            "fails":            None, 
            "miss":             None            
        }

    def set_filename(self, filename):
        self._metrics_data["filename"] = filename
    
    def store_metrics(self, real, predicted, missing, added, fails, correct):
        self._list_metrics.append(self._metrics_data)
        self._metrics_data["total"] = len(real)
        total_fails = len(fails) + len(missing)
        self._metrics_data["total_correct"] = len(correct)
        self._metrics_data["total_fails"] = total_fails
        self._metrics_data["precision"] = 100 - (total_fails * 100 / len(real))

        self._metrics_data["ground_truth"] = real
        self._metrics_data["generated"] = predicted
        self._metrics_data["added"] = added
        self._metrics_data["missing"] = missing
        self._metrics_data["fails"] = fails
        self._metrics_data["correct"] = correct
    
    def compute_classification_agreement(self, gt, gen):
        missing = []
        added = []
        missclassified = []
        good_clasified = []


        # Added: 
        subset = gen.loc[np.isin(gen[0], gt[0])==False]
        for text in subset[0]:
            added.append(text)


        for row in gt.iterrows():
            
            text, label = row[1]
            subset = gen.loc[gen[0]==text]

            # Si no se encuentra la key es que no ha respetado el csv de entrada
            if subset.empty:
                missing.append(text)

            else:
                result = subset.loc[subset[1]==label]

                # Si result esta vacio es que había almenos una coincidencia para 
                # text en el dataframe pero esta no tenia el label que buscamos
                if result.empty:
                    missclassified.append(text)
                
                else:
                    # Se elimina el primer resutado (podría haber varios pero deberian ir en orden)
                    finded_index = result.index[0]
                    gen = gen.drop(finded_index)

                    good_clasified.append(text)

        return missing, added, missclassified, good_clasified

    def get_array_ground_truth(self, filename):
        df = pd.read_csv(os.path.join('data', 'processed', 'anon', filename), sep='\t', header=None)

        aux = df[1].values
        df[1] = list(map(lambda x: x.split(' ')[0], aux))

        # result = [{row[2]: row[1]} for _, row in df.iterrows()]
        result = df[[2,1]]
        result.columns = [0,1]
        
        return result

    def calculate_classification_metrics(self,  filename, df_generated):

        df_ground_truth = self.get_array_ground_truth(filename.replace('.txt', '.ann'))

        # df_generated = [{row[0]: row[1]} for _, row in df.iterrows()]

        missing, added, missclassified, good_clasified = self.compute_classification_agreement(df_ground_truth, df_generated)
        self.store_metrics(df_ground_truth, df_generated, missing, added, missclassified, good_clasified)
    
    def save_metrics(self):
        df = pd.DataFrame(self._list_metrics)
        df.to_csv(f"data/metrics/{self._name_model}_metrics.csv")