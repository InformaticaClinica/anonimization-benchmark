import time

# Models
from llm import LLMContext
#from llm import BigLlamaModel, BigLlama3_1Model, ChatGPTModel, ChatGPTminiModel
from llm import BigLlamaModel, BigLlama3_1Model
from llm import Haiku3Model, Sonet3Model, OpusModel
from llm import BigMistralModel, Sonet3_5Model, SmallLlamaModel
from metrics import Metrics
import os
import re
import json

PATH = './data/carmen/'

# TODO: Move this function on utils.py
def read_text(filename=None):
    with open(filename, 'r') as archivo:
        return archivo.read()

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

def post_processing_replace_text(text, json_strings):
    # Merge all dictionaries parsed from JSON strings
    merged_dict = {}
    
    for json_string in json_strings:
        dictionary = json.loads(json_string)  # Parse JSON string to dictionary
        merged_dict.update(dictionary)
    
    def replacement_match(match):
        keyword = match.group(1)
        return merged_dict.get(keyword, keyword)
    
    # Using a regular expression to find texts inside [**]
    modified_text = re.sub(r'\[\*\*(.*?)\*\*\]', replacement_match, text)
    return modified_text


def anonimized_loop(llm, name_model):
    data = {}
    start_time = time.time()
    counter = 0
    context = LLMContext(llm)
    metrics = Metrics(name_model)
    for filename in sorted(os.listdir(f'{PATH}txt/replaced/')):
        metrics.set_filename(filename)
        try:
            data["system"] = read_text("prompts/system_prompt1.txt")
            data["user"] = read_text(f'{PATH}txt/replaced/{filename}')
            ground_truth = read_text(f'{PATH}txt/masked/{filename}')
            # First iteration
            text_generated = context.generate_response(data)
            # Second iteration
            data["system"] = read_text("prompts/system_prompt2_beta.txt")
            data["user"] = text_generated
            context = LLMContext(llm)
            dictionary = context.generate_response(data)
            #dictionary = json.load(dictionary)
            text_generated = post_processing_replace_text(text_generated, dictionary)
            print(text_generated)
            #metrics.calculate(ground_truth, text_generated)
            #metrics.store_metrics()
            #store_text(text_generated, filename, name_model)
            counter += 1
            if counter >= 100:
                break
            if counter % 10 == 0:
                print(counter)
        except Exception as e:
            print(e)
    metrics.save_metrics()
    save_time_to_file(name_model, start_time)

def call_models():
    anonimized_loop(SmallLlamaModel(),  "small_llama")
    # anonimized_loop(BigMistralModel(),  "big_mistral_model",   data)
    # anonimized_loop(BigLlama3_1Model(), "big_llama_3_1_model", data)
    # anonimized_loop(Sonet3_5Model(),    "sonet_3_5_model",     data)
    # anonimized_loop(BigLlamaModel(),    "big_llama_model",     data)
    # # anonimized_loop(ChatGPTModel(),     "chatgpt_model",       data)
    # # anonimized_loop(ChatGPTminiModel(),     "chatgpt_mini_model",       data)
    # anonimized_loop(Sonet3Model(),      "sonet_3_model",       data)
    # anonimized_loop(Haiku3Model(),      "haiku_3_model",       data)
    # anonimized_loop(OpusModel(),        "opus_model",          data)


def main():
    call_models()

if __name__ == "__main__":
    main()
