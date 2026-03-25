# prompt.py
import json

def build_prompt(schema: dict, context: dict = None) -> str:
    # Handle the old format {"fields": ["a", "b"]} and the new format {"key": "value", ...}
    if "fields" in schema:
        if isinstance(schema["fields"], list):
            schema_dict = {f: "" for f in schema["fields"]}
        elif isinstance(schema["fields"], dict):
            schema_dict = schema["fields"]
        else:
            schema_dict = schema
    else:
        schema_dict = schema

    field_list_items = []
    for k, v in schema_dict.items():
        if v:
            field_list_items.append(f"- {k}: {v}")
        else:
            field_list_items.append(f"- {k}")
            
    field_list = "\n".join(field_list_items)
    
    context_str = ""
    if context:
        context_str = f"\n\nHere is the information already extracted so far:\n{json.dumps(context, indent=2)}\n\nPlease Update this information or fill in new fields based on new input."
    
    # Dynamic example matching actual requested fields
    example_output = {k: "<extracted value or null>" for k in schema_dict.keys()}

    return f"""You are a precise medical scribe assistant analyzing a doctor-patient conversation.

Extract ONLY the following fields into JSON:
{field_list}{context_str}

RULES:
1. OUTPUT: Return ONLY valid JSON. No markdown, no preamble. 
2. LANGUAGE: Non-English input MUST be translated to English in the JSON values.
3. SPEAKER IDENTIFICATION:
   - DOCTOR: Asks questions, uses medical terms ("dosage", "diagnosis"), gives instructions, makes assessments.
   - PATIENT: Describes symptoms, provides history, answers questions.
4. CONTEXT: If a speaker's identity is unclear, use the 'Extracted so far' context to infer who they are.
5. CLEANING: Ignore filler words and hallucinations (e.g. "thanks for watching").

Output Structure:
{json.dumps(example_output, indent=2)}"""