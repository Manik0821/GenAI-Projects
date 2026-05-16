"""Transforms raw culinary map text into structured restaurant JSON records."""

# 1.1: Define the file_path to the text file
file_path = "California-Culinary-Map.txt"   # change this if your file is in another folder

# 1.2: Open the text file
with open(file_path, "r", encoding="utf-8") as file:
    data = file.read()

# 1.3: Print the first 100 characters of the restaurant data
# print(data[:100])

# 2.1: Split the restaurant paragraphs into list
restaurant_list = data.split("\n\n")   # split by blank lines

# 2.2: Since the first item is the dataset name, we remove it
restaurant_list = restaurant_list[1:]

# 2.3: Print out the number of restaurants we have
# print("Number of restaurants:", len(restaurant_list))

# 2.4: Print out the first item to have a closer look
# print("\nFirst restaurant entry:\n")
# print(restaurant_list[0])

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("NVIDIA_API_KEY1"),
    base_url="https://integrate.api.nvidia.com/v1"
)

def llm_model(system_msg, prompt_txt):
    # system_msg: the system message given to the LLM
    # prompt_txt: the user prompt

    model_id = "meta/llama-3.1-70b-instruct"  # NVIDIA model

    # 1.1: Define the model → (just a string here, not an object)
    
    # 1.2: Define the messages
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt_txt}
    ]

    # 1.3: Get the final response
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0.5,
        max_tokens=300
    )

    return response.choices[0].message.content

import time

def safe_llm_call(system_msg, prompt_txt, retries=3):
    for i in range(retries):
        try:
            return llm_model(system_msg, prompt_txt)
        except Exception:
            time.sleep(2)
    return "Failed after retries"

EXAMPLE_RESTAURANT_PARAGRAPH = restaurant_list[1] #use the second restaurant paragraph as the example
EXAMPLE_OUTPUT = """
    {{
    "name": "Mar de Cortez",
    "location": "Santa Monica",
    "type": "casual taqueria",
    "food_style": "Baja-style seafood",
    "rating": 4.2,
    "price_range": 1,
    "signatures": [
        "beer-battered snapper tacos",
        "zesty octopus ceviche"
    ],
    "vibe": "salt-air energy",
    "environment": "a premier sun-drenched spot for open-air dining near the pier."
    "shortcomings": []
    }}
"""

def restaurant_data_structure_prompt_generation(restaurant_paragraph):
    base_system_msg = f"""
You are an information extraction assistant.

Your job is to convert restaurant descriptions into structured JSON format.

Rules:
- Output ONLY valid JSON
- Do NOT add explanations or extra text
- Follow the exact schema shown in the example
- If any field is missing, use null or an empty list
- rating must be a number
- price_range must be a number (e.g., $, $$ → 1, 2, etc.)
- signatures must be a list of dishes
"""

    base_user_prompt = f"""
Task:
Extract structured information from the restaurant description and return JSON.

Fields required:
- name
- location
- type
- food_style
- rating
- price_range
- signatures (list)
- vibe
- environment
- shortcomings (list)

Restaurant description:
{restaurant_paragraph}

Example:
Input Restaurant Description:
{EXAMPLE_RESTAURANT_PARAGRAPH}

Output:
{EXAMPLE_OUTPUT}
"""
    
    return base_system_msg, base_user_prompt

restaurant_paragraph = restaurant_list[0]
base_system_msg, base_user_prompt = restaurant_data_structure_prompt_generation(restaurant_paragraph=restaurant_paragraph)

test_response = llm_model(system_msg=base_system_msg, prompt_txt=base_user_prompt)

# Validation
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional

### 3.1. Define the schema
class Restaurant(BaseModel):
    name: str
    location: str
    type: str
    food_style: str
    rating: Optional[float] = None
    price_range: Optional[int] = None
    signatures: List[str] = Field(default_factory=list)
    vibe: Optional[str] = None
    environment: str
    shortcomings: List[str] = Field(default_factory=list)


### 3.2. Use the validation method to validate the test_response from the unit test
# try:
#     restaurant_data = Restaurant.model_validate_json(test_response)
#     print(f"Success! Validated: {restaurant_data.name}")
# except ValidationError as e:
#     print(f"Validation failed: {e.json()}")

def JSON_auto_repair_prompts(candidate_json_output, error_message):
    auto_repair_system_msg = """
You are a JSON repair assistant.

Your job is to fix invalid JSON and return valid JSON.

Rules:
- Output ONLY valid JSON
- Do NOT add explanations or extra text
- Preserve the original meaning and data
- Fix syntax issues (missing commas, quotes, brackets, etc.)
- If a value is missing, use null or an empty list
- Ensure proper JSON formatting
"""

    auto_repair_prompt = f"""
The following JSON is invalid.

Error message:
{error_message}

Invalid JSON:
{candidate_json_output}

Task:
Fix the JSON and return a valid JSON object.
"""

    return auto_repair_system_msg, auto_repair_prompt


import json

structured_restaurant_lists = []

for i, restaurant_paragraph in enumerate(restaurant_list):
    
    # 2.1: Produce initial output
    system_msg, user_prompt = restaurant_data_structure_prompt_generation(restaurant_paragraph)
    output = safe_llm_call(system_msg, user_prompt)

    # 2.2: Validation + Auto Correction loop
    max_attempts = 3
    attempt = 0

    while attempt < max_attempts:
        try:
            parsed_output = json.loads(output)  # try parsing
            break  # success → exit loop

        except Exception as e:
            attempt += 1
            
            # generate repair prompt
            repair_sys, repair_prompt = JSON_auto_repair_prompts(output, str(e))
            output = safe_llm_call(repair_sys, repair_prompt)

    # 2.3: Append final result
    try:
        structured_restaurant_lists.append(json.loads(output))
    except:
        print(f"Skipping entry {i} due to repeated JSON errors")
        continue

    # Progress indicator
    if (i + 1) % 20 == 0:
        print(f'{i+1} out of {len(restaurant_list)} is done')

# Completion message
print('ALL DONE!!')


# ---------------------------
# 9. Add itemId
# ---------------------------
for i, restaurant in enumerate(structured_restaurant_lists):
    restaurant["itemId"] = 1000001 + i


# ---------------------------
# 10. Save to file
# ---------------------------
with open("structured_restaurants.json", "w") as f:
    json.dump(structured_restaurant_lists, f, indent=2)


# ---------------------------
# 11. Print sample (50th item)
# ---------------------------
if len(structured_restaurant_lists) >= 50:
    print(structured_restaurant_lists[49])
else:
    print("Less than 50 items")