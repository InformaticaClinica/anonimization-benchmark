import time

# Models
from llm import LLMContext
#from llm import BigLlamaModel, BigLlama3_1Model, ChatGPTModel, ChatGPTminiModel
from llm import BigLlamaModel, BigLlama3_1Model
from llm import Haiku3Model, Sonet3Model, OpusModel
from llm import BigMistralModel, Sonet3_5Model, SmallLlamaModel
from metrics import Metrics, MetricsDict
import os
import re
import json
from io import StringIO
import csv
import pandas as pd 

PATH = './data/carmen/'

# TODO: Move this function on utils.py
def read_text(filename=None):
    with open(filename, 'r') as archivo:
        return archivo.read()


def output_to_csv(data):

    if data.strip() == 'None':
        return None

    # Crear un objeto StringIO a partir del string
    datos_io = StringIO(data)

    # Crear un lector CSV con el delimitador de coma y el caracter de comillas dobles
    lector_csv = csv.reader(datos_io, delimiter=',', quotechar='"', skipinitialspace=True)

    filas  = []

    # Iterar sobre cada fila y agregarla a la lista
    for fila in lector_csv:
        filas.append(fila)
    
    return pd.DataFrame(filas)

def save_time_to_file(block, start_time):
    end_time_1 = time.time()
    execution_time = end_time_1 - start_time
    filename = f"./data/metrics/time/{block}.txt"
    with open(filename, 'a') as file:
        file.write(f"Execution time of {block}: {execution_time} seconds\n")

def store_text(text, filename, name_model):
    folder_path = f"./data/anon/raw/{name_model}"
    file_path = os.path.join(folder_path, f"{filename}")
    os.makedirs(folder_path, exist_ok=True)
    with open(file_path, 'w') as file:
        file.write(text)

def post_processing_replace_text(text, dictionary_list):
    combined_dictionary = {}
    for d in dictionary_list:
            combined_dictionary.update(d)
    def replacement_match(match):
        keyword = match.group(1)
        return combined_dictionary.get(keyword, keyword)
    
    modified_text = re.sub(r'\[\*\*(.*?)\*\*\]', replacement_match, text)
    return modified_text


def first_iteration(metrics, filename, llm, name_model):
    context = LLMContext(llm)
    data = {}
    data["system"] = read_text("prompts/system_prompt1.txt")
    data["user"] = read_text(f'{PATH}txt/replaced/{filename}')
    metrics.set_filename(filename)
    ground_truth = read_text(f'{PATH}/masked/{filename}')
    text_generated = context.generate_response(data)
    metrics.calculate(ground_truth, text_generated)
    metrics.store_metrics()
    store_text(text_generated, filename, "first/" + name_model)
    return metrics, text_generated

# TODO: implement logic when there is no labels to classify (input_text = '')
def second_iteration(metrics_second, text_generated, llm, name_model, filename):
    metrics_second.set_filename(filename)
    prompt_filename = 'prompts/system_prompt2_beta.txt'

    # Process text_generated to create a list of labels
    labels = re.findall(r'\[\*\*(.*?)\*\*\]', text_generated)
    labels = list(map(lambda x: f'"{x}"', labels))
    input_text = '\n'.join(labels)

    prompt = {"system": read_text(prompt_filename), "user": input_text}
    context = LLMContext(llm)
    output = context.generate_response(prompt)

    df = output_to_csv(output)

    # metrics_second.calculate_classification_metrics(
    #     filename,
    #     dictionary
    # )

    directorio = f"./data/json/{name_model}"
    os.makedirs(directorio, exist_ok=True)
    output_path = os.path.join(directorio, f"{filename}.csv")
    df.to_csv(output_path)

    return df

def third_iteration(metrics_thrid, text_generated, dictionary, name_model, filename):
    metrics_thrid.set_filename(filename)
    ground_truth = read_text(f'{PATH}txt/masked/{filename}')
    text_generated = post_processing_replace_text(text_generated, dictionary)
    metrics_thrid.calculate(
        ground_truth, 
        text_generated, 
        classification_bool = False
        )
    metrics_thrid.store_metrics()
    store_text(text_generated, filename, "thrid/" + name_model+"_3rd")
    return metrics_thrid

def anonimized_loop(llm, name_model):
    start_time = time.time()
    counter = 0
    metrics = Metrics(name_model)
    metrics_second = MetricsDict(name_model+"_2rd")
    metrics_thrid = Metrics(name_model+"_3rd")
    for filename in sorted(os.listdir(f'{PATH}txt/replaced/')):
        try:
            metrics, text_generated = first_iteration(metrics, filename, llm, name_model)
            dictionary = second_iteration(metrics_second, text_generated, llm, name_model, filename)
            metrics_thrid = third_iteration(metrics_thrid, text_generated, dictionary, name_model, filename)
            counter += 1
        except Exception as e:
            print(e)
    metrics.save_metrics()
    metrics_second.save_metrics()
    metrics_thrid.save_metrics()
    save_time_to_file(name_model, start_time)

def call_models():
    anonimized_loop(SmallLlamaModel(),  "small_llama")
    anonimized_loop(BigMistralModel(),  "big_mistral_model")
    anonimized_loop(BigLlama3_1Model(), "big_llama_3_1_model")
    anonimized_loop(Sonet3_5Model(),    "sonet_3_5_model")
    anonimized_loop(BigLlamaModel(),    "big_llama_model")
    # anonimized_loop(ChatGPTModel(),     "chatgpt_model")
    # anonimized_loop(ChatGPTminiModel(),     "chatgpt_mini_model")
    anonimized_loop(Sonet3Model(),      "sonet_3_model")
    anonimized_loop(Haiku3Model(),      "haiku_3_model")
    anonimized_loop(OpusModel(),        "opus_model")


def main():
    call_models()

if __name__ == "__main__":
    main()
