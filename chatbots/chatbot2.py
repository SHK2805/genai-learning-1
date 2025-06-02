from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
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
    return RunnableWithMessageHistory(model, get_session_history)

def chat_bot1(input_config, name=None, hobby=None) :
    bot = get_chatbot()
    session_id = input_config.get("configurable", {}).get("session_id", "default_session")
    response_bot = bot.invoke(
    [HumanMessage(content="Hello, how are you?")],
    config=input_config
    )


    print(f"{session_id} Response:", response_bot.content)
    response_bot = bot.invoke(
    [
        AIMessage(content=response_bot.content),
        HumanMessage(content="What is my name?"),
    ],
    config=input_config
    )
    print(f"{session_id} Response:", response_bot.content)

    response_bot = bot.invoke(
        [
            AIMessage(content=response_bot.content),
            HumanMessage(content=f"OK. My name is {name}."),
        ],
        config=input_config
    )

    response_bot = bot.invoke(
        [
            AIMessage(content=response_bot.content),
            HumanMessage(content="What is my name?"),
        ],
        config=input_config
    )
    print(f"{session_id} Response:", response_bot.content)

    response_bot = bot.invoke(
        [
            AIMessage(content=response_bot.content),
            HumanMessage(content=f"My hobby is {hobby}."),
        ],
        config=input_config
    )
    print(f"{session_id} Response:", response_bot.content)

    response_bot = bot.invoke(
        [
            AIMessage(content=response_bot.content),
            HumanMessage(content="What is my hobby?"),
        ],
        config=input_config
    )
    print(f"{session_id} Response:", response_bot.content)
    return response_bot.content

def call_chat_bot1():
    config1 = {"configurable": {"session_id": "session1"}}
    chat_bot1(config1, name="John Doe", hobby="reading")

    config2 = {"configurable": {"session_id": "session2"}}
    chat_bot1(config2, name="Jack Smith", hobby="gaming")

if __name__ == "__main__":
    call_chat_bot1()
    print("Chatbot interactions completed.")







