# Models
from llm import LLMContext
from llm import BigLlamaModel, BigLlama3_1Model, ChatGPTModel
from llm import Haiku3Model, Sonet3Model, OpusModel
from llm import BigMistralModel, Sonet3_5Model, SmallLlamaModel
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