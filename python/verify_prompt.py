import json
from prompt import build_prompt

def test_build_prompt():
    schema = {"fields": ["Chief complaint", "duration"]}
    
    # Test case 1: No instructions or knowledgebase
    prompt = build_prompt(schema)
    assert "Chief complaint" in prompt
    assert "KNOWLEDGE BASE" not in prompt
    assert "ADDITIONAL INSTRUCTIONS" not in prompt
    print("Test Case 1 Passed: Basic prompt works.")

    # Test case 2: With instructions
    instructions = "Suggest some lab orders for this."
    prompt = build_prompt(schema, instructions=instructions)
    assert "ADDITIONAL INSTRUCTIONS:" in prompt
    assert instructions in prompt
    print("Test Case 2 Passed: Instructions added correctly.")

    # Test case 3: With knowledgebase
    knowledgebase = "Aspirin: 100mg, Ibuprofen: 200mg"
    prompt = build_prompt(schema, knowledgebase=knowledgebase)
    assert "KNOWLEDGE BASE (Priority Reference for Medications/Lab Orders):" in prompt
    assert knowledgebase in prompt
    assert "IMPORTANT: Use the above knowledge base" in prompt
    print("Test Case 3 Passed: Knowledgebase added correctly.")

    # Test case 4: Both instructions and knowledgebase
    prompt = build_prompt(schema, instructions=instructions, knowledgebase=knowledgebase)
    assert "ADDITIONAL INSTRUCTIONS:" in prompt
    assert "KNOWLEDGE BASE" in prompt
    print("Test Case 4 Passed: Both added correctly.")

    # Test case 5: Different schema format
    schema_dict = {"Chief complaint": "The main reason for visit"}
    prompt = build_prompt(schema_dict)
    assert "The main reason for visit" in prompt
    print("Test Case 5 Passed: Different schema format works.")

if __name__ == "__main__":
    try:
        test_build_prompt()
        print("\nAll unit tests for build_prompt passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
