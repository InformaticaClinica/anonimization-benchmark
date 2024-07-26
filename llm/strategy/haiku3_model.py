import boto3
import json
from .llm_strategy import LLMStrategy
from llm.prompt_handling.haiku_prompt_handler import HaikuPromptHandler

class Haiku3Model(LLMStrategy):
    def __init__(self):
        super().__init__()
        self._model_id = "anthropic.claude-3-haiku-20240307-v1:0"
        print(self._model_id)
        self._client = boto3.client('bedrock-runtime', region_name='eu-west-3')
        self._prompt_handler = HaikuPromptHandler()
        self._top_k = 50
        self._accept = 'application/json'
        self._contentType = 'application/json'


    def create_body(self, prompt) -> str:
        return json.dumps({
            "prompt": prompt,
            "max_tokens_to_sample":self._max_gen_len,
            "temperature":self._temperature,
            "top_p":self._top_p,
            "top_k":self._top_k
        })

    def invoke_model(self, body: dict) -> str:
        response = self._client.invoke_model(
            modelId=self._model_id,
            body=body,
            accept=self._accept,
            contentType=self._contentType
        )
        response = json.loads(response.get('body').read())
        return response["completion"]

    def generate_prompt(self, model_prompt: dict) -> str:
        prompt = self._prompt_handler.format_prompt(model_prompt)
        body = self.create_body(prompt)
        return self.invoke_model(body)