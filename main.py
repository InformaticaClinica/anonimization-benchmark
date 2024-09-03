# MODELS
from llm import LLMContext, BigLlamaModel, BigLlama3_1Model, SmallLlamaModel, BigMistralModel, Haiku3Model, Sonet3Model, OpusModel, Sonet3_5Model

import os


def anonimized_loop(llm, name_model, data):
    counter = 0
    context = LLMContext(llm)
    list_data  = []
    for filename in sorted(os.listdir(PATH)):
        print(filename)


def main():
    print("Hello world")


if __name__ == "__main__":
    main()