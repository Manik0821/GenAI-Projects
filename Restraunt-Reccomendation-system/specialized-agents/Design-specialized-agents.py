import os
from typing import List, Dict, Any
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from define_agents import user_profile_agent_config, rag_retriever_agent_config, food_trend_analyst_config, food_style_expert_config,  nutrition_expert_config, recommendation_expert_config

# ================================
# Load .env from parent directory
# ================================
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# ================================
# Debug: check key
# ================================
api_key = os.getenv("NVIDIA_API_KEY1")

if not api_key:
    raise ValueError("❌ NVIDIA_API_KEY1 not found in .env")

print("✅ NVIDIA key loaded")

# ================================
# Initialize NVIDIA OpenAI client
# ================================
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

MODEL = "meta/llama-3.1-70b-instruct"

# ================================
# Test agent config import
# ================================
# print(f"Role: {user_profile_agent_config['role']}")
# print(f"\nGoal: {user_profile_agent_config['goal']}")
# print(f"\nBackstory: {user_profile_agent_config['backstory']}")

# print(f"Role: {recommendation_expert_config['role']}")
# print(f"\nGoal: {recommendation_expert_config['goal']}")
# print(f"\nBackstory: {recommendation_expert_config['backstory']}")

# print(f"Role: {nutrition_expert_config['role']}")
# print(f"\nGoal: {nutrition_expert_config['goal']}")
# print(f"\nBackstory: {nutrition_expert_config['backstory']}")

# print(f"Role: {food_trend_analyst_config['role']}")
# print(f"\nGoal: {food_trend_analyst_config['goal']}")
# print(f"\nBackstory: {food_trend_analyst_config['backstory']}")

# print(f"Role: {rag_retriever_agent_config['role']}")
# print(f"\nGoal: {rag_retriever_agent_config['goal']}")
# print(f"\nBackstory: {rag_retriever_agent_config['backstory']}")

def create_agent_prompt(agent_config: Dict[str, str]) -> str:
    """Create a system prompt for an agent based on its configuration."""
    prompt = f"""You are a {agent_config['role']}.
    
Your goal: {agent_config['goal']}

Your background: {agent_config['backstory']}

Always respond in a professional, helpful manner that reflects your expertise.
"""
    return prompt

# Create agent prompts
user_profile_prompt = create_agent_prompt(user_profile_agent_config)
rag_retriever_prompt = create_agent_prompt(rag_retriever_agent_config)
food_trend_prompt = create_agent_prompt(food_trend_analyst_config)
food_style_prompt = create_agent_prompt(food_style_expert_config)
nutrition_prompt = create_agent_prompt(nutrition_expert_config)
recommendation_prompt = create_agent_prompt(recommendation_expert_config)

print("Agent prompts created successfully!")

task_generate_profile = {
    "description": """Analyze the user's restaurant visit history and social media posts to create a comprehensive profile.
    Extract the following information:
    - Favorite cuisines and cuisine categories
    - Dietary restrictions or preferences (vegetarian, vegan, gluten-free, etc.)
    - Preferred dining occasions (casual, fine dining, quick bites)
    - Price sensitivity
    - Adventurousness (comfort food lover vs. culinary explorer)
    - Flavor preferences (spicy, sweet, savory, etc.)
    - Frequency of dining out
    
    Provide specific examples from the user's history to support each insight.""",
    
    "expected_output": """A structured user profile in JSON format with keys: 
    favorite_cuisines, dietary_restrictions, dining_occasions, price_range, 
    adventurousness_score (1-10), flavor_preferences, dining_frequency.
    Include a summary paragraph explaining the user's dining personality.""",
    
    "agent": "User Profile Generator"
}

print(f"Task: {task_generate_profile['description'][:100]}...")
print(f"\nExpected Output: {task_generate_profile['expected_output'][:100]}...")
print(f"\nAgent: {task_generate_profile['agent']}")

task_retrieve_candidates = {
    "description": """Based on the user profile, query the vector database to retrieve:
    - Top 20 restaurants that match the user's preferences
    - Top 20 recipes that align with their taste and dietary needs
    
    Use similarity search with the user's favorite cuisines and flavor preferences as the query.
    Apply filters for dietary restrictions, price range, and location (if provided).
    Ensure diversity in the results—don't retrieve 20 Italian restaurants if the user likes multiple cuisines.""",
    
    "expected_output": """Two lists in JSON format:
    - restaurants: Array of restaurant objects with fields: name, cuisine_type, price_range, rating, description
    - recipes: Array of recipe objects with fields: name, cuisine_type, difficulty, prep_time, ingredients, description""",
    
    "agent": "RAG Retriever"
}

print(f"Task: {task_retrieve_candidates['description'][:100]}...")
print(f"\nExpected Output: {task_retrieve_candidates['expected_output'][:100]}...")
print(f"\nAgent: {task_retrieve_candidates['agent']}")

task_analyze_trends = {
    "description": """Analyze current food trends relevant to the retrieved restaurants and recipes.
    Identify:
    - Trending ingredients or techniques in the retrieved items
    - Popular dining concepts or restaurant types
    - Emerging culinary movements that align with the user's interests
    - Seasonal trends or timely food moments
    
    Provide context on why these trends matter and how they enhance the recommendations.""",
    
    "expected_output": """A trends analysis with:
    - List of 3-5 relevant trends with descriptions
    - Explanation of how each trend relates to the user's profile
    - Suggestions for which restaurants or recipes align with these trends""",
    
    "agent": "Food Trend Analyst"
}

## Type your answer here

task_analyze_food_styles = {
    "description": """Analyze the cuisine types, regional variations, cooking methods, and flavor profiles of the retrieved restaurants and recipes.

    Identify:
    - Dominant cuisine categories present in the results (e.g., Italian, Japanese, Indian regional styles)
    - Regional distinctions within cuisines (e.g., Sichuan vs Cantonese, Neapolitan vs Roman)
    - Key cooking techniques used (e.g., grilling, steaming, fermentation, slow-cooking)
    - Flavor profiles (e.g., spicy, umami-rich, sweet, acidic, creamy, smoky)
    - Relationships between dishes and their cultural or culinary origins

    Use this analysis to clearly explain how each food item reflects its culinary identity and what makes it distinctive.""",

    "expected_output": """A structured food style analysis including:
    - Breakdown of cuisine types and regional variations found in results
    - Summary of key cooking techniques observed
    - Description of dominant flavor profiles across items
    - Insights linking food styles to user preferences and query intent
    - Optional grouping of items by similar culinary characteristics""",

    "agent": "Food Style Expert"
}

task_evaluate_nutrition = {
    "description": """Evaluate the nutritional aspects of the retrieved restaurants and recipes.
    Check for:
    - Alignment with dietary restrictions (vegetarian, vegan, gluten-free, etc.)
    - Potential allergens or ingredients to avoid
    - Nutritional balance (protein, vegetables, whole grains)
    - Healthfulness relative to the user's goals
    
    Flag any items that don't meet the user's dietary needs and explain why.""",
    
    "expected_output": """A nutrition evaluation with:
    - List of items that meet all dietary restrictions
    - Items flagged for potential concerns (allergens, restrictions)
    - Nutritional highlights (high protein, vegetable-rich, etc.)
    - Overall assessment of how well the options support the user's health goals""",
    
    "agent": "Nutrition Expert"
}

task_generate_recommendations = {
    "description": """Synthesize insights from all previous agents to generate final recommendations.
    Create:
    - Top 5 restaurant recommendations with detailed explanations
    - Top 5 recipe recommendations with detailed explanations
    
    For each recommendation, explain:
    - Why it matches the user's profile
    - How it aligns with current trends (if applicable)
    - What makes it a great fit in terms of food style
    - Any nutritional benefits or considerations
    
    Write in an engaging, enthusiastic tone that makes the user excited to try these options.""",
    
    "expected_output": """A recommendations report with:
    - restaurants: Array of 5 restaurant recommendations with name, description, and detailed reasoning
    - recipes: Array of 5 recipe recommendations with name, description, and detailed reasoning
    - Each recommendation should include a personalized explanation (2-3 sentences) of why it's a great match""",
    
    "agent": "Recommendation Expert"
}

print(f"Task: {task_generate_recommendations['description'][:100]}...")
print(f"\nExpected Output: {task_generate_recommendations['expected_output'][:100]}...")
print(f"\nAgent: {task_generate_recommendations['agent']}")

# Sample user data
sample_user_data = """
Restaurant Visit History:
- Visited "Spice Route" (Indian, $$) 5 times in the last 3 months
- Visited "Green Earth Cafe" (Vegan, $) 3 times
- Visited "Ramen House" (Japanese, $$) 2 times
- Visited "Taco Fiesta" (Mexican, $) 4 times

Social Media Posts:
- "Loving this spicy curry at Spice Route! 🌶️🔥"
- "Trying to eat more plant-based meals. This vegan bowl is delicious!"
- "Best ramen I've had in ages. The broth is perfection."
- "Late night tacos are the best tacos 🌮"
"""


def test_agent(agent_prompt: str, user_input: str) -> str:
    """Test an agent by sending it a sample input."""
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.7,
        messages=[
            {"role": "system", "content": agent_prompt},
            {"role": "user", "content": user_input}
        ]
    )
    return response.choices[0].message.content

class LLMClient:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NVIDIA_API_KEY1")
        )
        self.model = "meta/llama-3.1-70b-instruct"

    def run(self, system, user):
        res = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
        )
        return res.choices[0].message.content

# Test the User Profile Generator
# Note: This will only work if you have set your OpenAI API key
try:
    profile_result = test_agent(
        agent_prompt=user_profile_prompt,
        user_input=f"Analyze this user data and create a profile:\n\n{sample_user_data}"
    )
    print("=" * 80)
    print("USER PROFILE GENERATED")
    print("=" * 80)
    print(profile_result)
except Exception as e:
    print(f"Note: Agent testing requires a valid OpenAI API key. Error: {e}")
    print("\nYou have successfully designed all agents and tasks.")
    print("In Lesson 2, you will integrate these agents into a working system.")

# Create a summary of all agents and their tasks
agents_summary = [
    {"agent": "User Profile Generator", "task": "Generate User Profile"},
    {"agent": "RAG Retriever", "task": "Retrieve Relevant Restaurants and Recipes"},
    {"agent": "Food Trend Analyst", "task": "Analyze Food Trends"},
    {"agent": "Food Style Expert", "task": "Analyze Food Styles"},
    {"agent": "Nutrition Expert", "task": "Evaluate Nutrition and Dietary Fit"},
    {"agent": "Recommendation Expert", "task": "Generate Final Recommendations"}
]

print("=" * 80)
print("MULTI-AGENT SYSTEM SUMMARY")
print("=" * 80)
for i, item in enumerate(agents_summary, 1):
    print(f"{i}. {item['agent']:30} → {item['task']}")