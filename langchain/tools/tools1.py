from typing import List

from langchain_core.tools import create_retriever_tool

import langchain
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client
from langchain_experimental.agents import AgentType


from config.env_manager import langchain_api_key

langchain.verbose = False

client = Client(api_key=langchain_api_key())
prompt = client.pull_prompt("hwchase17/openai-functions-agent", include_model=True)

def wikipedia_tool():
    """Create a Wikipedia query tool."""
    wikipedia_api = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
    return WikipediaQueryRun(api_wrapper=wikipedia_api)

def arxiv_tool():
    """Create an Arxiv query tool."""
    arxiv_api = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
    return ArxivQueryRun(api_wrapper=arxiv_api)

def retriever_tool():
    loader = WebBaseLoader("https://en.wikipedia.org/wiki/Seven_Wonders_of_the_Ancient_World")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    tool = create_retriever_tool(retriever,
                                           "pyramids_search",
                                           "Search for information about the Seven Wonders of the Ancient World using a retriever.")
    return tool

tool_wiki = wikipedia_tool()
tool_arxiv = arxiv_tool()
tool_retriever = retriever_tool()


def get_tools() -> List:
    """Get a list of tools."""
    return [tool_wiki, tool_arxiv, tool_retriever]

from config.set_config import Config
# Load environment variables from .env file
# Initialize Config
config = Config()
if config.set():
    print("Environment variables set")
else:
    print("Environment variables NOT set")

llm = ChatOpenAI(model="gpt-4o")

create_openai_tools_agent



