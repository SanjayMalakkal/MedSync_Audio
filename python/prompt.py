# prompt.py
import json

def build_prompt(schema: dict, context: dict = None) -> str:
    fields = schema.get("fields", [])
    field_list = "\n".join([f"- {f}" for f in fields])
    
    context_str = ""
    if context:
        context_str = f"\n\nHere is the information already extracted so far:\n{json.dumps(context, indent=2)}\n\nPlease Update this information or fill in new fields based on the new audio. Do not lose existing information unless the user corrects it."
    
    # Dynamic example matching actual requested fields
    example_output = {f: "<extracted value or null>" for f in fields}

    return f"""You are a precise medical information extraction system.

Extract ONLY the following fields from the audio conversation:
{field_list}{context_str}

Rules:
- Return ONLY valid JSON, no markdown, no explanation
- Use null (not "null") for missing fields
- Ignore filler words (um, ah, like, you know)
- Normalize phone numbers to digits only

Output exactly this structure:
{json.dumps(example_output, indent=2)}"""