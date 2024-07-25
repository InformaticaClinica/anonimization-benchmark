import boto3
import json
from .llm_strategy import LLMStrategy
from llm.prompt_handling.llama3_prompt_handler import Llama3PromptHandler

class BigLlamaModel(LLMStrategy):
    def __init__(self):
        super().__init__()
        self._model_id = "meta.llama3-70b-instruct-v1:0"
        self._client = boto3.client('bedrock-runtime', region_name='eu-west-2')
        self._prompt_handler = Llama3PromptHandler()
        self._temperature = 0.1
        self._max_gen_len = 2000
        self._top_p = 0.9

    def create_body(self, prompt: str) -> str:
        return json.dumps({
            "prompt": prompt,
            "max_gen_len":self._max_gen_len,
            "temperature":self._temperature,
            "top_p":self._top_p
        })
    
    def invoke_model(self, body: str) -> str:
        response = self._client.invoke_model(
            body=body,
            modelId=self._model_id
        )
        return json.loads(response.get('body').read())["generation"]


    def generate_prompt(self, model_prompt: dict) -> str:
        prompt = self._prompt_handler.format_prompt(model_prompt)
        body = self.create_body(prompt)
        return self.invoke_model(body)