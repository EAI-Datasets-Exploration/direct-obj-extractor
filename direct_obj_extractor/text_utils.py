import json
import re

# Function to Construct Prompts
def format_prompt(text):
    return f"""
    Extract the direct objects and verbs from the following sentence and return them in JSON format:

    Sentence: "{text}"

    Output format:
    {{
        "direct_objects": ["object1", "object2", ...],
        "verbs": ["verb1", "verb2", ...]
    }}
    """

# Extract JSON from LLM Response
def extract_json(text):
    """
    Extracts and parses the final JSON block from the LLM response.

    Returns:
        direct_objects (list): List of extracted direct objects.
        verbs (list): List of extracted verbs.
    """
    # Find all JSON occurrences in the response
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    
    if matches:
        # Use only the last detected JSON block (most likely the final answer)
        json_text = matches[-1]  

        # Clean up formatting issues
        json_text = json_text.replace('```json', '').replace('```', '')  # Remove Markdown code block
        json_text = json_text.replace('“', '"').replace('”', '"')  # Fix curly quotes
        json_text = json_text.replace('\n', '').strip()  # Remove newlines and spaces

        # Check for Placeholder JSON Output (detect and skip)
        if "object1" in json_text or "verb1" in json_text or "..." in json_text:
            print(f"Skipping template JSON: {json_text}")
            return [], []

        # Attempt to parse JSON
        try:
            extracted_json = json.loads(json_text)
            direct_objects = extracted_json.get("direct_objects", [])
            verbs = extracted_json.get("verbs", [])

            # Ensure lists (handle cases where model outputs a string instead of a list)
            direct_objects = direct_objects if isinstance(direct_objects, list) else [direct_objects]
            verbs = verbs if isinstance(verbs, list) else [verbs]

            return direct_objects, verbs

        except json.JSONDecodeError:
            print(f"JSON decoding error: {json_text}")  # Debugging info
            return [], []  # Return empty lists if JSON parsing fails

    return [], []  # Return empty lists if no JSON is found

# Function to remove determiners for sorting
def clean_determiners(text):
    return re.sub(r"^(the|a|an)\s+", "", text, flags=re.IGNORECASE)  # Remove "the", "a", "an" at the start
