# Query RAG and Testing Framework

This project provides a framework for performing Retrieval-Augmented Generation (RAG) queries using the `langchain` library, with support for embedding-based search and response evaluation. It also includes test cases to validate query responses using a Language Learning Model (LLM).

## Project Structure

### Files

- `query_rag_script.py`: Implements the RAG query process using the Chroma vector store and Ollama LLM.
- `test_query_rag.py`: Contains test cases and validation functions for ensuring accurate query results.
- `get_embedding_function.py`: Provides the embedding function used in the RAG pipeline.
- `README.md`: Documentation for understanding and using the project.

### Key Components

#### `query_rag_script.py`
- **Purpose**: Retrieves and processes data using Chroma, formats context-based prompts, and generates responses using the Ollama LLM.
- **Functions**:
  - `process_query`: Handles embedding-based similarity search and response generation.
  - `main`: Entry point for running queries via command-line.

#### `test_query_rag.py`
- **Purpose**: Contains test cases to validate query responses by comparing them against expected outputs.
- **Functions**:
  - `test_monopoly_rules`: Tests Monopoly rules.
  - `test_ticket_to_ride_rules`: Tests Ticket to Ride rules.
  - `validate_query_result`: Compares actual and expected responses using an LLM for evaluation.

## Installation

### Prerequisites
- Python 3.8+
- `langchain`, `argparse`, and other required libraries (install via `pip install -r requirements.txt`)

### Setup
1. Clone the repository.
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your PDF files for processing in the `data` directory.

## Usage

### Query Execution
Run the RAG query process from the command line:
```bash
python query_rag_script.py "<Your Query Text>"
```
Example:
```bash
python query_rag_script.py "How much money does a player start with in Monopoly?"
```

### Testing Queries
To validate queries:
1. Run `test_query_rag.py` using a test framework like `pytest`:
   ```bash
   pytest test_query_rag.py
   ```
2. Each test checks whether the actual response matches the expected response.

## Examples

### Query Example
Input:
```text
Your question 1?
```
Output:
```text
Response: Your expected answer
Sources: ["data/your_uploaded_pdf.pdf:2:1"]
```

### Test Output
If a test passes, the output will be highlighted in green:
```text
Response: true
```
If a test fails, the output will be highlighted in red:
```text
Response: false
```

## Notes
- Update the `CHROMA_DATABASE_DIRECTORY` and `EVALUATION_PROMPT_TEMPLATE` constants as needed for your dataset and evaluation criteria.
- Customize the test cases in `test_query_rag.py` to validate additional rules or query formats.

## Contribution
Contributions to improve the framework or add features are welcome. Please follow these steps:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request with a detailed description of changes.


