# chain: prompt | llm | output_parser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

from config.set_config import Config
# config
config = Config()
if config.set():
    print("Environment variables set")
else:
    print("Environment variables NOT set")

# message
message_capital_of_france = "What is the capital of France?"
output_parser = JsonOutputParser()
# prompt, llm, output_parser
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{input}\n",
    input_variables=["input"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
)

llm = ChatOpenAI(model="gpt-3.5-turbo")

chain = prompt | llm | output_parser

result = chain.invoke({"input": message_capital_of_france})
# print the result
print(result)