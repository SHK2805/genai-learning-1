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


if __name__ == "__main__":

    print("Chatbot interactions completed.")







