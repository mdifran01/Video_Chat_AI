from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from core.tools import search_knowledge_base


llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


agent = create_agent(
    model=llm,
    tools=[search_knowledge_base],

    system_prompt="""
You are a helpful RAG assistant.

You answer questions using the user's knowledge base.

When a question requires information from the
knowledge base, use the search_knowledge_base tool.

Use the retrieved information to answer the question.

Do not invent information.

After answering, provide the source information
returned by the tool.

For YouTube or audio sources, include the timestamp
when available.

Keep the answer clear and concise.
"""
)