# imports
# web loader
# read from a web page
# split
# embeddings
# store into a vector store
# llm
# prompt
# retriever
# document chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from config.set_config import Config
# Load environment variables from .env file
# Initialize Config
config = Config()
if config.set():
    print("Environment variables set")
else:
    print("Environment variables NOT set")

loader = WebBaseLoader("https://en.wikipedia.org/wiki/Seven_Wonders_of_the_Ancient_World")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

vector_store = FAISS.from_documents(documents, embeddings)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})


llm = ChatOpenAI(model="o4-mini")
system_message: str = "You are a helpful assistant that answers questions based on the provided context only."
prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    ("human", "{context}\n\nQuestion: {input}"),
])
doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
retrieval_chain = create_retrieval_chain(retriever, doc_chain)

query = "What are the Seven Wonders of the Ancient World?"
response = retrieval_chain.invoke({"input": query})
print(response["answer"])
