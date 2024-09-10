from llm import LLMContext
#from llm import BigLlamaModel, BigLlama3_1Model, ChatGPTModel
from llm import BigLlamaModel, BigLlama3_1Model
from llm import Haiku3Model, Sonet3Model, OpusModel
from llm import BigMistralModel, Sonet3_5Model, SmallLlamaModel
from llm import ChatGPTModel, ChatGPTminiModel
from metrics import Metrics
import os

PATH = './data/processed/'


def read_text(filename=None):
    with open(filename, 'r',encoding='utf-8') as archivo:
        return archivo.read()

def anonimized_loop(llm, name_model, data):
    counter = 0
    context = LLMContext(llm)
    metrics = Metrics(name_model)
    for filename in sorted(os.listdir(f'{PATH}txt/')):
        #print(filename)
        metrics.set_filename(filename)
        try:
            data["user"] = read_text(f'{PATH}txt/{filename}')
            ground_truth = read_text(f'{PATH}masked/{filename}')
            text_generated = context.generate_response(data)
            metrics.calculate(ground_truth, text_generated)
            metrics.store_metrics()
            counter += 1
            print(counter)
            if counter >= 100:
                break
        except Exception as e:
            print(e)
    metrics.save_metrics()

def anonimized_loop_2(llm, name_model, data):
    counter = 0
    context = LLMContext(llm)
    metrics = Metrics(name_model)
    for filename in sorted(os.listdir(f'{PATH}txt/')):
        metrics.set_filename(filename)
        try:
            data["user"] = read_text(f'{PATH}txt/{filename}')
            ground_truth = read_text(f'{PATH}masked/{filename}')
            text_generated = context.generate_response(data)
            metrics.calculate()
            metrics.store_metrics()
            counter += 1
            if counter >= 100:
                break
        except Exception as e:
            print(e)
    metrics.save_metrics()

def call_models(data):
    # anonimized_loop(BigLlama3_1Model(), "big_llama_3_1_model", data)
    # anonimized_loop(BigMistralModel(),  "big_mistral_model",   data)
    # anonimized_loop(SmallLlamaModel(),  "small_llama",         data)
    # anonimized_loop(Sonet3_5Model(),    "sonet_3_5_model",     data)
    # anonimized_loop(BigLlamaModel(),    "big_llama_model",     data)
    anonimized_loop(ChatGPTModel(),     "chatgpt_model",       data)
    anonimized_loop(ChatGPTminiModel(),     "chatgpt_mini_model",       data)
    # anonimized_loop(Sonet3Model(),      "sonet_3_model",       data)
    # anonimized_loop(Haiku3Model(),      "haiku_3_model",       data)
    #anonimized_loop(OpusModel(),        "opus_model",          data)


def main():
    data = {}
    data["system"] = read_text("prompts/system_prompt1.txt")
    call_models(data)

if __name__ == "__main__":
    main()
