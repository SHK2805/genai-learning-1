import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from config.set_config import Config
# Load environment variables from .env file
config = Config()
if config.set():
    print("Environment variables set")
else:
    print("Environment variables NOT set")

# test text
texts = ["How do I get a replacement Medicare card?",
        "What is the monthly premium for Medicare Part B?",
        "How do I terminate my Medicare Part B (medical insurance)?",
        "How do I sign up for Medicare?",
        "Can I sign up for Medicare Part B if I am working and have health insurance through an employer?",
        "How do I sign up for Medicare Part B if I already have Part A?",
        "What are Medicare late enrollment penalties?",
        "What is Medicare and who can get it?",
        "How can I get help with my Medicare Part A and Part B premiums?",
        "What are the different parts of Medicare?",
        "Will my Medicare premiums be higher because of my higher income?",
        "What is TRICARE ?",
        "Should I sign up for Medicare Part B if I have Veterans' Benefits?"]

# OpenAI Embeddings
# Create an instance of OpenAIEmbeddings with the specified model
openai_embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
# Embed the text using the OpenAIEmbeddings instance
openai_embedding = openai_embeddings.embed_documents(texts)
# Print the resulting embedding
print("Embedding:", openai_embedding)
print("Embedding length (Number of Vectors):", len(openai_embedding))

print("**********************************************************")

# HuggingFace Embeddings
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Embed the text using the HuggingFaceEmbeddings instance
hf_embedding = hf_embeddings.embed_documents(texts)
# Print the resulting embedding
print("HuggingFace Embedding:", hf_embedding)
print("HuggingFace Embedding length (Number of Vectors):", len(hf_embedding))

print("**********************************************************")

