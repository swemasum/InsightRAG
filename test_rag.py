from query_data import query_rag
from langchain_community.llms.ollama import Ollama

# Define the evaluation prompt template
EVALUATION_PROMPT_TEMPLATE = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""

def test_monopoly_rules():
    """
    Test case for verifying the starting money rule in Monopoly.
    """
    assert validate_query_result(
        question="How much total money does a player start with in Monopoly? (Answer with the number only)",
        expected_response="$1500",
    )

def test_ticket_to_ride_rules():
    """
    Test case for verifying the longest continuous train rule in Ticket to Ride.
    """
    assert validate_query_result(
        question="How many points does the longest continuous train get in Ticket to Ride? (Answer with the number only)",
        expected_response="10 points",
    )

def validate_query_result(question: str, expected_response: str) -> bool:
    """
    Validate the query result against the expected response using an evaluation LLM.

    Args:
        question: The input query text.
        expected_response: The expected response to validate against.

    Returns:
        bool: True if the actual response matches the expected response, False otherwise.
    """
    # Fetch the actual response from the query RAG function
    actual_response = query_rag(question)

    # Format the evaluation prompt
    evaluation_prompt = EVALUATION_PROMPT_TEMPLATE.format(
        expected_response=expected_response, actual_response=actual_response
    )

    # Invoke the evaluation model
    evaluation_model = Ollama(model="mistral")
    evaluation_result = evaluation_model.invoke(evaluation_prompt).strip().lower()

    # Log the evaluation process
    print(evaluation_prompt)

    # Determine the result and print feedback
    if "true" in evaluation_result:
        print("\033[92m" + f"Response: {evaluation_result}" + "\033[0m")  # Green for correct
        return True
    elif "false" in evaluation_result:
        print("\033[91m" + f"Response: {evaluation_result}" + "\033[0m")  # Red for incorrect
        return False
    else:
        raise ValueError("Invalid evaluation result. Expected 'true' or 'false'.")
