import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

load_dotenv()

# Load environment variables from .env file
# openai API key
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Langchain
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGSMITH_TRACING_V2'] = os.getenv('LANGSMITH_TRACING_V2')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')

message_capital_of_france = "What is the capital of France?"
# chat openai
llm = ChatOpenAI(model="o4-mini")
# chat prompt template
prompt_capital_of_france = ChatPromptTemplate(
    messages=[
        ("system", "You are a helpful assistant."),
        ("user", "{input}"),
    ]
)
# chain
chain_capital_of_france = prompt_capital_of_france | llm
# invoke
response_capital_of_france = chain_capital_of_france.invoke(
    {"input": message_capital_of_france}
)
# print the response
print(response_capital_of_france.content)
