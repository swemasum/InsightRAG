import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function

# Define constants for Chroma database path and prompt template
CHROMA_DATABASE_DIRECTORY = "chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    """
    Main function to handle command-line arguments and query processing.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Query the Chroma database using RAG.")
    parser.add_argument("query_text", type=str, help="The input query text.")
    args = parser.parse_args()

    # Process the query
    process_query(args.query_text)

def process_query(query_text: str):
    """
    Perform a RAG (Retrieval-Augmented Generation) query on the database.

    Args:
        query_text: The input query text.
    """
    # Initialize the Chroma database with the embedding function
    embedding_function = get_embedding_function()
    chroma_db = Chroma(persist_directory=CHROMA_DATABASE_DIRECTORY, embedding_function=embedding_function)

    # Search for similar documents in the database
    search_results = chroma_db.similarity_search_with_score(query_text, k=5)

    # Build the context from retrieved documents
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in search_results])

    # Format the prompt with the retrieved context and query
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Initialize the LLM model and generate a response
    llm_model = Ollama(model="mistral")
    response_text = llm_model.invoke(prompt)

    # Extract sources and format the response
    sources = [doc.metadata.get("id", "Unknown") for doc, _ in search_results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"

    # Output the response
    print(formatted_response)
    return response_text

if __name__ == "__main__":
    main()
