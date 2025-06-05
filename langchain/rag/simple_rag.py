from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.set_config import Config
from constants import (text_chunk_size,
                       text_chunk_overlap,
                       openai_embeddings_name,
                       fiass_similarity_search_k, openai_model_name)



# Load environment variables from .env file
# Initialize Config
config = Config()
if config.set():
    print("Environment variables set")
else:
    print("Environment variables NOT set")

"""
steps
Import the necessary libraries
Get the URL of the web page
Import necessary libraries
Create the WebBaseLoader instance to load the web page
Load the documents from the web page using the web loader
Split the loaded documents into smaller chunks for processing
Convert the chunks into embeddings for further processing or analysis
Store the embeddings into a vector store or use them for similarity search
Query the vector store for relevant information based on a user query using the embeddings and similarity search
The results are then used as a context for the LLM to generate a response for answer the query.
Create a prompt template for the LLM to format the input
Create the LLM Model instance
Create the stuff documents chain to process the context and question
Give the context and question to the LLM to generate the answer
"""


# Define the URL to load
web_url: str = "https://en.wikipedia.org/wiki/Seven_Wonders_of_the_Ancient_World"

# Create a WebBaseLoader instance to load the web page
loader = WebBaseLoader(web_url)

# Load the documents from the web page
document = loader.load()

# split the documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=text_chunk_size, chunk_overlap=text_chunk_overlap)
documents = splitter.split_documents(document)

# Convert the chunks into embeddings
embeddings = OpenAIEmbeddings(model=openai_embeddings_name)

# Store the embeddings into a vector store or use them for similarity search
vector_store = FAISS.from_documents(documents, embeddings)

# Query the vector store for relevant information based on a user query
query:str = "What are the Seven Wonders of the Ancient World?"
results = vector_store.similarity_search(query, k=fiass_similarity_search_k)
# Print the results
"""
for result in results:
    print(result.page_content)
    print("\n---\n")
"""

# Prepare the context and question for the LLM
context = results # "\n\n".join([result.page_content for result in results])
question = query

# create the LLM instance
llm = ChatOpenAI(model=openai_model_name)

# Create a prompt template for the LLM
system_message = ("You are a helpful assistant that answers questions based on the provided context."
                  "Answer only with the information provided in the context"
                  "If you do not know say I do not know.")

prompt = ChatPromptTemplate(
    [
        ("system", system_message),
        ("human", "{context}\n\n{question}"),
    ]
)
# Create the stuff documents chain to process the context and question
chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt,
    document_variable_name="context",
)
# Give the context and question to the LLM to generate the answer
response = chain.invoke({
    "context": context,
    "question": question
})
# Print the response
print(response)
















