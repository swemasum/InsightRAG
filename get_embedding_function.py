from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings

def initialize_embedding_provider():
    """
    Initializes and returns the embedding provider.

    By default, the function uses AWS Bedrock embeddings with a specified
    credentials profile and region. Uncomment the other provider to use
    Ollama embeddings instead.

    Returns:
        Embedding provider instance.
    """
    # Configure Bedrock embeddings with AWS credentials and region
    bedrock_embeddings_provider = BedrockEmbeddings(
        credentials_profile_name="default",  # AWS credentials profile name
        region_name="us-east-1"              # AWS region for the service
    )

    # Uncomment the following lines to use Ollama embeddings instead
    # ollama_embeddings_provider = OllamaEmbeddings(model="nomic-embed-text")
    
    return bedrock_embeddings_provider
    # return ollama_embeddings_provider
