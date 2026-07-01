"""
Processing pipeline for recipe and user-review image enrichment.

Outputs:
- augmented_food_recipe.json
- augmented_user_review.json
"""

import ast
import base64
import json
import os
import time
import warnings
import zipfile
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential


warnings.filterwarnings("ignore")
load_dotenv()


BASE_DIR = Path(__file__).resolve().parent

RECIPES_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/hpTjb6liKBLVHQK0UgMi5A/Recipes.json"
REVIEWS_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/fQUs9wQ6aB6ts6fmkD2V2w/Synthetic-User-Reviews.json"
IMAGES_ZIP_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5_Rr6ohviItzucyWk6nkrw/synthetic-recipe-images.zip"

RECIPES_FILE = BASE_DIR / "Recipes.json"
REVIEWS_FILE = BASE_DIR / "Synthetic-User-Reviews.json"
IMAGES_ZIP_FILE = BASE_DIR / "synthetic-recipe-images.zip"
RECIPE_IMAGE_DIR = BASE_DIR / "synthetic_recipe_images"

AUGMENTED_RECIPE_FILE = BASE_DIR / "augmented_food_recipe.json"
AUGMENTED_REVIEW_FILE = BASE_DIR / "augmented_user_review.json"

TEMP_REVIEW_IMAGE = BASE_DIR / "review_image_placeholder.jpg"


# ---------------------------
# Download helpers
# ---------------------------

def download_file(url: str, output_path: Path) -> None:
    if output_path.exists():
        print(f"Already exists: {output_path.name}")
        return

    print(f"Downloading {output_path.name}...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(response.content)

    print(f"Downloaded: {output_path.name}")


def setup_files() -> None:
    download_file(RECIPES_URL, RECIPES_FILE)
    download_file(REVIEWS_URL, REVIEWS_FILE)
    download_file(IMAGES_ZIP_URL, IMAGES_ZIP_FILE)

    if not RECIPE_IMAGE_DIR.exists():
        print("Extracting recipe images...")
        with zipfile.ZipFile(IMAGES_ZIP_FILE, "r") as zip_ref:
            zip_ref.extractall(BASE_DIR)
        print("Images extracted.")
    else:
        print("Recipe image folder already exists.")


# ---------------------------
# API key failover
# ---------------------------

def get_nvidia_api_keys() -> list[str]:
    keys = [
        os.getenv("NVIDIA_API_KEY"),
        os.getenv("NVIDIA_API_KEY1"),
        os.getenv("NVIDIA_API_KEY2"),
        os.getenv("NVIDIA_API_KEY3"),
    ]

    clean_keys = []
    for key in keys:
        if key and key.strip() and key.strip() not in clean_keys:
            clean_keys.append(key.strip())

    if not clean_keys:
        raise RuntimeError(
            "No NVIDIA API key found. Add NVIDIA_API_KEY or NVIDIA_API_KEY1/2/3 in your .env file."
        )

    return clean_keys


def get_openai_client(api_key: str) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url="https://integrate.api.nvidia.com/v1",
    )


def is_retryable_llm_error(error: Exception) -> bool:
    text = str(error).lower()
    retry_signals = [
        "429",
        "rate",
        "quota",
        "timeout",
        "temporarily",
        "connection",
        "server",
        "500",
        "502",
        "503",
        "504",
    ]
    return any(signal in text for signal in retry_signals)


# ---------------------------
# Vision LLM
# ---------------------------

def vision_llm(system_msg: str, prompt_txt: str, image_path: Path) -> str:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    ext = image_path.suffix.lower().replace(".", "")
    if ext == "jpg":
        ext = "jpeg"

    with open(image_path, "rb") as img_file:
        image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

    messages = [
        {
            "role": "system",
            "content": system_msg,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_txt,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{ext};base64,{image_base64}",
                    },
                },
            ],
        },
    ]

    last_error = None

    for api_key in get_nvidia_api_keys():
        try:
            client = get_openai_client(api_key)

            response = client.chat.completions.create(
                model="meta/llama-3.2-90b-vision-instruct",
                messages=messages,
                max_tokens=300,
                temperature=0.5,
            )

            return response.choices[0].message.content.strip()

        except Exception as error:
            last_error = error
            print(f"Vision LLM failed with one key: {error}")

            if not is_retryable_llm_error(error):
                continue

            time.sleep(2)

    raise RuntimeError(f"Vision LLM failed with all keys. Last error: {last_error}")


# ---------------------------
# Prompt templates
# ---------------------------

def image_caption_prompt_template(food_name: str) -> tuple[str, str]:
    system_msg = f"""
You are a food image captioning assistant.

Your task:
- Describe the food shown in the image
- Focus specifically on the dish: "{food_name}"
- Keep the description clear and concise
- Mention visible ingredients, textures, and presentation
- Do NOT add unrelated details
"""

    prompt_txt = f"""
Describe the dish "{food_name}" shown in this image.

Include:
- Key ingredients you can see
- Appearance including color and texture
- Presentation style

Keep it under 3 to 4 sentences.
"""

    return system_msg, prompt_txt


def review_context_image_caption_prompt_template(review_text: str) -> tuple[str, str]:
    system_msg = """
You are a multimodal food review analyst.

Your task:
- Look at the food image and the user review text
- Generate a caption that combines visual information and user sentiment
- Be accurate and only describe what is visible in the image
- Use the review to understand taste, quality, and experience
- Do NOT hallucinate ingredients not visible
"""

    prompt_txt = f"""
Generate a detailed caption for the food image.

User review:
{review_text}

Instructions:
- Describe the dish in the image
- Incorporate user sentiment when relevant
- Mention appearance, texture, and presentation
- Keep it 3 to 5 sentences
"""

    return system_msg, prompt_txt


# ---------------------------
# Recipe image processing
# ---------------------------

def find_recipe_image(recipe_id: int | str) -> Path | None:
    possible_dirs = [
        BASE_DIR / "recipe_images",
        BASE_DIR / "synthetic_recipe_images",
    ]

    for folder in possible_dirs:
        for ext in ["png", "jpg", "jpeg"]:
            image_path = folder / f"recipe{recipe_id}.{ext}"
            if image_path.exists():
                return image_path

    return None

def process_recipe_images(recipe_data: list[dict]) -> list[dict]:
    print("\nProcessing recipe images...")

    for index, recipe in enumerate(recipe_data):
        if (index + 1) % 5 == 0:
            print(f"{index + 1} out of {len(recipe_data)} recipes processed")

        food_name = recipe.get("name", "Unknown dish")
        recipe_id = recipe.get("id")

        image_path = find_recipe_image(recipe_id)

        if image_path is None:
            print(f"Image not found for recipe id: {recipe_id}")
            recipe["image_description"] = None
            continue

        try:
            system_msg, prompt_txt = image_caption_prompt_template(food_name)
            caption = vision_llm(system_msg, prompt_txt, image_path)
            recipe["image_description"] = caption

        except Exception as error:
            print(f"Failed recipe id {recipe_id}: {error}")
            recipe["image_description"] = None

    print("Recipe image processing completed.")
    return recipe_data


# ---------------------------
# Review image processing
# ---------------------------

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10))
def get_data_with_retry(url: str) -> requests.Response:
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return response


def parse_review_images(images_value) -> list[str]:
    if not images_value:
        return []

    if isinstance(images_value, list):
        return images_value

    if isinstance(images_value, str):
        try:
            parsed = ast.literal_eval(images_value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return []

    return []


def save_review_image(url: str, output_path: Path) -> bool:
    try:
        response = get_data_with_retry(url)

        with open(output_path, "wb") as img_file:
            img_file.write(response.content)

        Image.open(output_path).verify()
        return True

    except Exception as error:
        print(f"Failed to download image {url}: {error}")
        return False


def process_review_images(user_review_data: list[dict]) -> list[dict]:
    print("\nProcessing review images...")

    for index, review in enumerate(user_review_data):
        if (index + 1) % 5 == 0:
            print(f"{index + 1} out of {len(user_review_data)} reviews processed")

        review_images = parse_review_images(review.get("images"))
        review_text = review.get("text", "")

        review_image_captions = []

        for image_url in review_images:
            if not save_review_image(image_url, TEMP_REVIEW_IMAGE):
                continue

            try:
                system_msg, prompt_txt = review_context_image_caption_prompt_template(review_text)

                caption = vision_llm(
                    system_msg=system_msg,
                    prompt_txt=prompt_txt,
                    image_path=TEMP_REVIEW_IMAGE,
                )

                review_image_captions.append(caption)

            except Exception as error:
                print(f"Failed caption generation for review {index}: {error}")

        review["image_captions"] = review_image_captions

    print("Review image processing completed.")
    return user_review_data


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    setup_files()

    with open(RECIPES_FILE, "r", encoding="utf-8") as f:
        recipe_data = json.load(f)

    print("\nFirst recipe sample:")
    for key, value in recipe_data[0].items():
        print(f"{key} ({type(value).__name__}): {value}")

    recipe_data = process_recipe_images(recipe_data)

    with open(AUGMENTED_RECIPE_FILE, "w", encoding="utf-8") as f:
        json.dump(recipe_data, f, indent=4, ensure_ascii=False)

    print(f"Saved: {AUGMENTED_RECIPE_FILE}")

    with open(REVIEWS_FILE, "r", encoding="utf-8") as f:
        user_review_data = json.load(f)

    print("\nFirst review sample:")
    for key, value in user_review_data[0].items():
        print(f"{key} ({type(value).__name__}): {value}")

    user_review_data = process_review_images(user_review_data)

    with open(AUGMENTED_REVIEW_FILE, "w", encoding="utf-8") as f:
        json.dump(user_review_data, f, indent=4, ensure_ascii=False)

    print(f"Saved: {AUGMENTED_REVIEW_FILE}")
    print("\nALL DONE!")


if __name__ == "__main__":
    main()