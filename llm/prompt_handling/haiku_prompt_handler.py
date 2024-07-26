from .prompt_handler import PromptHandler

class HaikuPromptHandler(PromptHandler):
    def format_prompt(self, data: dict):
        return f"""\n\nHuman:[INST]{data["system"]}


                {data["user"]}
            
            \n\nAssistant:"""