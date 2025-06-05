from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


from config.set_config import Config
from constants import openai_model_name

# Load environment variables from .env file
# Initialize Config
config = Config()
if config.set():
    print("Environment variables set")
else:
    print("Environment variables NOT set")

message_capital_of_france = "What is the capital of France?"

# chat openai
llm = ChatOpenAI(model=openai_model_name)

# chat prompt template
prompt_capital_of_france = ChatPromptTemplate(
    messages=[
        ("system", "You are a helpful assistant."),
        ("user", "{input}"),
    ]
)

# output parser
output_parser = StrOutputParser()

# chain
chain_capital_of_france = prompt_capital_of_france | llm | output_parser

# invoke
response_capital_of_france = chain_capital_of_france.invoke(
    {"input": message_capital_of_france}
)

# print the response
print(response_capital_of_france)
