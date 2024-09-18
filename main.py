import time

# Models
from llm import LLMContext
#from llm import BigLlamaModel, BigLlama3_1Model, ChatGPTModel, ChatGPTminiModel
from llm import BigLlamaModel, BigLlama3_1Model
from llm import Haiku3Model, Sonet3Model, OpusModel
from llm import BigMistralModel, Sonet3_5Model, SmallLlamaModel
from metrics import Metrics
import os

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

def anonimized_loop(llm, name_model, data):
    start_time = time.time()
    counter = 0
    context = LLMContext(llm)
    metrics = Metrics(name_model)
    for filename in sorted(os.listdir(f'{PATH}txt/replaced/')):
        print(filename)
        metrics.set_filename(filename)
        try:
            data["user"] = read_text(f'{PATH}txt/replaced/{filename}')
            ground_truth = read_text(f'{PATH}txt/masked/{filename}')
            # First iteration
            text_generated = context.generate_response(data)
            # Second iteration
            # TODO: Handle system prompt because of performance
            data["system"] = read_text("prompts/system_prompt2.txt")
            data["user"] = text_generated
            text_generated = context.generate_response(data)
            metrics.calculate(ground_truth, text_generated)
            metrics.store_metrics()
            store_text(text_generated, filename, name_model)
            counter += 1
            if counter >= 100:
                break
            if counter % 10 == 0:
                print(counter)
        except Exception as e:
            print(e)
    metrics.save_metrics()
    save_time_to_file(name_model, start_time)

def call_models(data):
    anonimized_loop(SmallLlamaModel(),  "small_llama",         data)
    anonimized_loop(BigMistralModel(),  "big_mistral_model",   data)
    anonimized_loop(BigLlama3_1Model(), "big_llama_3_1_model", data)
    anonimized_loop(Sonet3_5Model(),    "sonet_3_5_model",     data)
    anonimized_loop(BigLlamaModel(),    "big_llama_model",     data)
    # anonimized_loop(ChatGPTModel(),     "chatgpt_model",       data)
    # anonimized_loop(ChatGPTminiModel(),     "chatgpt_mini_model",       data)
    anonimized_loop(Sonet3Model(),      "sonet_3_model",       data)
    anonimized_loop(Haiku3Model(),      "haiku_3_model",       data)
    anonimized_loop(OpusModel(),        "opus_model",          data)


def main():
    data = {}
    data["system"] = read_text("prompts/system_prompt1.txt")
    call_models(data)

if __name__ == "__main__":
    main()