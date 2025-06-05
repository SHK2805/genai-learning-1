from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq

from config.set_config import Config
from constants import gorq_model_name

config = Config()
if config.set():
    print("Environment variables set")
else:
    print("Environment variables NOT set")

model = ChatGroq(model=gorq_model_name)
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get the chat message history for a session."""
    if session_id not in store:
        print(f"Creating new chat message history for session: {session_id}")
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def get_chatbot() -> RunnableWithMessageHistory:
    """Get a chatbot instance with message history for a session."""
    return RunnableWithMessageHistory(chain, get_session_history, input_messages_key="messages")

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="You are a helpful assistant. Answer all questions to the best of your ability."),
        MessagesPlaceholder(variable_name="messages")
    ]
)

chain = prompt | model
response = chain.invoke({"messages" : [HumanMessage(content="Hello, how are you?")]})
print("Response:", response.content)


config1 = {"configurable": {"session_id": "session1"}}
bot = get_chatbot()
response = bot.invoke({
    "messages": [
        AIMessage(content=response.content),
        HumanMessage(content="My name is Bob and I am a Data Scientist")]
    },
    config=config1
)

response = bot.invoke({
    "messages": [
        AIMessage(content=response.content),
        HumanMessage(content="I forgot, what is my name?")]
    },
    config=config1
)

print("Response:", response.content)

if __name__ == "__main__":

    print("Chatbot interactions completed.")







