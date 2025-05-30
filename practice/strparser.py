from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from config.set_config import Config

# config
config = Config()
if config.set():
    print("Environment variables set")
else:
    print("Environment variables NOT set")

# message
# prompt, llm, output_parser
prompt = ChatPromptTemplate(
    messages=[
        ("system", "You are a helpful assistant."),
        ("user", "{input}"),
    ]
)
llm = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser
result = chain.invoke({"input": "What is the capital of France?"})
print(result)

