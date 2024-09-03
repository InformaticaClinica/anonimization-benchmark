# MODELS
from llm.context.llm_context import LLMContext
from llm.strategy.big_llama_model import BigLlamaModel
from llm.strategy.big_llama3_1_model import BigLlama3_1Model
from llm.strategy.small_llama_model import SmallLlamaModel
from llm.strategy.big_mistral_model import BigMistralModel
from llm.strategy.haiku3_model import Haiku3Model
from llm.strategy.sonet3_model import Sonet3Model
from llm.strategy.opus3_model import OpusModel
from llm.strategy.sonet3_5_model import Sonet3_5Model
#from llm.strategy.chatgpt_model import ChatGPTModel

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