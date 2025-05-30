from langchain_community.document_loaders import WebBaseLoader
from config.set_config import Config

# Load environment variables from .env file
# Initialize Config
config = Config()
if config.set():
    print("Environment variables set")
else:
    print("Environment variables NOT set")


web_url: str = "https://en.wikipedia.org/wiki/Seven_Wonders_of_the_Ancient_World"





