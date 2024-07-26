import boto3
import json
from .llm_strategy import LLMStrategy
from llm.prompt_handling.haiku_prompt_handler import SonetPromptHandler

class Sonet3Model(LLMStrategy):
    def __init__(self):
        super().__init__()
        self._model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
        self._client = boto3.client('bedrock-runtime', region_name='eu-west-2')
        self._prompt_handler = SonetPromptHandler()
        self._top_k = 50