from langchain_openai import ChatOpenAI
from langchain.agents import create_agent as lc_create_agent
from langchain_core.tools import tool


def create_agent(vector_store):
    """Create and return a LangChain agent with a document search tool."""
    llm = ChatOpenAI(temperature=0)

    @tool("document_search", description="Search enterprise documents")
    def search_docs(query: str) -> str:
        """Search enterprise documents for content relevant to the user query."""
        docs = vector_store.similarity_search(query, k=4)
        if not docs:
            return "No relevant documents found."
        return "\n\n".join(doc.page_content for doc in docs)

    agent = lc_create_agent(
        model=llm,
        tools=[search_docs],
        system_prompt="You are a helpful assistant that searches enterprise documents.",
    )

    return agent