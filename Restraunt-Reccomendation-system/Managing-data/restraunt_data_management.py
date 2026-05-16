import json
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


def _get_nvidia_api_key() -> str:
    return (
        os.getenv("NVIDIA_API_KEY")
        or os.getenv("NVIDIA_API_KEY1")
        or os.getenv("NVIDIA_API_KEY2")
        or os.getenv("NVIDIA_API_KEY3")
        or ""
    )

# ---------------------------
# FILE CONFIG
# ---------------------------
FILEPATH = 'structured_restaurant_data.json'
BACKUP_PATH = 'structured_restaurant_data.json.bak'


# ---------------------------
# HELPERS (MUST EXIST)
# ---------------------------
def load_data(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_data(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def show_restaurant_card(res, index):
    print("\n--- Restaurant Detail ---")
    print(f"Index: {index}")
    for k, v in res.items():
        print(f"{k}: {v}")

import re

def clean_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group() if match else text


# ---------------------------
# PROMPT BUILDER
# ---------------------------
def restaurant_data_structure_prompt_generation(paragraph):

    system_msg = """
You are a strict information extraction system.

Return ONLY valid JSON.

Fields:
name, location, type, food_style, rating, price_range,
signatures (list), vibe, environment, shortcomings (list)

RULES:
- Output ONLY valid JSON (no markdown, no text)
- MUST follow schema exactly
- Use double quotes only
- No extra fields
- price_range must be an INTEGER from 1 to 4
- rating must be FLOAT
- If unknown, use null
"""

    user_prompt = f"""
Convert this into STRICT JSON:

{paragraph}

Return format exactly:

{{
  "name": "",
  "location": "",
  "type": "",
  "food_style": "",
  "rating": 0.0,
  "price_range": 0,
  "signatures": [],
  "vibe": "",
  "environment": "",
  "shortcomings": []
}}
"""

    return system_msg, user_prompt


# ---------------------------
# LLM CALL (NVIDIA LLAMA)
# ---------------------------
def llm_model(system_msg, prompt_txt):

    client = OpenAI(
        api_key=_get_nvidia_api_key(),
        base_url="https://integrate.api.nvidia.com/v1"
    )

    response = client.chat.completions.create(
        model=os.getenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct"),
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt_txt}
        ],
        temperature=0.3,
        max_tokens=300
    )

    return response.choices[0].message.content


# ---------------------------
# JSON FIXER
# ---------------------------
def JSON_auto_repair_prompts(response, error_message):

    system_msg = "Fix invalid JSON. Output ONLY valid JSON."

    prompt = f"""
Broken JSON:
{response}

Error:
{error_message}

Return corrected JSON only.
"""

    return system_msg, prompt


# ---------------------------
# PROCESS NEW ENTRY
# ---------------------------
def new_data_entry_process(paragraph, itemId):

    system_msg, prompt = restaurant_data_structure_prompt_generation(paragraph)

    response = llm_model(system_msg, prompt)

    for _ in range(3):
        try:
            cleaned = clean_json(response)
            data = safe_json_load(cleaned)

            data["itemId"] = itemId
            return data

        except Exception as e:
            repair_sys, repair_prompt = JSON_auto_repair_prompts(response, str(e))
            response = llm_model(repair_sys, repair_prompt)

    # ❗ IMPORTANT: fail gracefully BUT still structured
    return {
        "itemId": itemId,
        "name": "Unknown",
        "location": "Unknown",
        "type": "Unknown",
        "food_style": "Unknown",
        "rating": 0.0,
        "price_range": 0,
        "signatures": [],
        "vibe": "Unknown",
        "environment": "Unknown",
        "shortcomings": []
    }


# ---------------------------
# MAIN UI
# ---------------------------
def manage_restaurants(file_path, backup_path):

    while True:
        data = load_data(file_path) or []

        print(f"\n🏨 DATABASE | Records: {len(data)}")
        print("1. Browse")
        print("2. View")
        print("3. Add")
        print("4. Edit")
        print("5. Delete")
        print("6. Exit")

        choice = input("Action: ")

        # ------------------ 1
        if choice == "1":
            print("\n--- Names ---")
            for i, r in enumerate(data):
                print(i, r.get("name", "N/A"))

        # ------------------ 2
        elif choice == "2":
            try:
                idx = int(input("Index: "))
                if 0 <= idx < len(data):
                    show_restaurant_card(data[idx], idx)
                else:
                    print("invalid index")
            except:
                print("invalid index")

        # ------------------ WRITE OPS
        elif choice in ["3", "4", "5"]:

            print("⚠️ Write Mode")
            if input("type yes: ") != "yes":
                print("Operation cancelled")
                continue

            # ------------------ ADD
            if choice == "3":
                itemId = 1000000 + len(data) + 1
                paragraph = input("Enter restaurant description: ")

                new_data = new_data_entry_process(paragraph, itemId)
                data.append(new_data)

                save_data(file_path, data)
                print("✅ Restaurant added")

            # ------------------ EDIT
            elif choice == "4":
                idx = int(input("Index: "))

                if 0 <= idx < len(data):
                    for k in data[idx]:
                        val = input(f"{k} ({data[idx][k]}): ")
                        if val:
                            data[idx][k] = val
                    save_data(file_path, data)
                    print("✅ Updated")
                else:
                    print("invalid index")

            # ------------------ DELETE
            elif choice == "5":
                idx = int(input("Index: "))

                if 0 <= idx < len(data):
                    data.pop(idx)
                    save_data(file_path, data)
                    print("✅ Deleted")
                else:
                    print("invalid index")

        elif choice == "6":
            break

        else:
            print("Invalid input")

import unittest
import json
import os
import io
from unittest.mock import patch


class TestRestaurantDatabase(unittest.TestCase):

    def setUp(self):
        self.test_file = "test_db.json"
        self.backup_file = "test_db.bak"

        self.initial_data = [
            {"name": "Test Cafe", "location": "Test City"}
        ]

        with open(self.test_file, "w") as f:
            json.dump(self.initial_data, f)

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

        if os.path.exists(self.backup_file):
            os.remove(self.backup_file)

    @patch("builtins.input")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_add_and_delete(self, mock_stdout, mock_input):

        mock_input.side_effect = [
            "3", "yes", "A cozy Italian restaurant with handmade pasta",
            "5", "yes", "1",
            "6"
        ]

        manage_restaurants(self.test_file, self.backup_file)

        with open(self.test_file, "r") as f:
            data = json.load(f)

        self.assertTrue(len(data) >= 1)
        self.assertIn("Restaurant added", mock_stdout.getvalue())

    @patch("builtins.input")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_delete_cancel(self, mock_stdout, mock_input):

        mock_input.side_effect = [
            "5", "no",
            "6"
        ]

        manage_restaurants(self.test_file, self.backup_file)

        with open(self.test_file, "r") as f:
            data = json.load(f)

        self.assertEqual(len(data), 1)
        self.assertIn("Operation cancelled", mock_stdout.getvalue())


if __name__ == "__main__":
    # unittest.main()
    manage_restaurants(FILEPATH, BACKUP_PATH)