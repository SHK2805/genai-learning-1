import os
from config.env_manager import *

# this gets the key-values from the .env file and sets the environment
class Config:
    def __init__(self):
        self.groq_api_key = groq_api_key()
        self.openai_api_key = openai_api_key()
        self.langchain_api_key = langchain_api_key()
        self.hf_api_key = hf_api_key()

    def set(self):
        try:
            os.environ['GROQ_API_KEY'] = self.groq_api_key
            os.environ['OPENAI_API_KEY'] = self.openai_api_key
            os.environ['LANGCHAIN_API_KEY'] = self.langchain_api_key
            os.environ['HF_TOKEN'] = self.hf_api_key
            """
            Tokenizers throwing warning "The current process just got forked, Disabling parallelism to avoid deadlocks.. 
            To disable this warning, please explicitly set TOKENIZERS_PARALLELISM=(true | false)"
            """
            os.environ['TOKENIZERS_PARALLELISM'] = "false"


        except Exception as e:
            print(f'Error setting environment variables: {e}')
            return False
        return True