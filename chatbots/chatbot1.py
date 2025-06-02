from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from config.set_config import Config
from constants import gorq_model_name

config = Config()
if config.set():
    print("Environment variables set")
else:
    print("Environment variables NOT set")

model = ChatGroq(model=gorq_model_name)

messages = [SystemMessage(content="You are an expert Data Scientist professor. Help the students with their questions."),
            HumanMessage(content="Hello, My name is John Doe."),
            AIMessage(content="Hello John Doe, how can I assist you today?"),
            HumanMessage(content="I am a Data Scientist.")]
result = model.invoke(messages)

messages.append(AIMessage(content=result.content))
messages.append(HumanMessage(content="What is the difference between supervised and unsupervised learning?"))
messages.append(HumanMessage(content="What is my name and what do I do?"))
result = model.invoke(messages)
print(result.content)
messages.append(AIMessage(content=result.content))
