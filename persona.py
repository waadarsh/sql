from dotenv import load_dotenv
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# Set your Mistral AI API key
load_dotenv('.env')

# Define the model
model = "mistral-small-latest"

# Initialize the Mistral Client
client = MistralClient()

# The system prompt to define the task
system_prompt = """
You are a helpful assistant that generates detailed persona data based on given user input. Each persona should include the fields: "Persona", "Description", "Keywords", "Visual_Textual_Contents", "Points_of_Focus", "Emotional_Drivers", "Demographics", and "Behavioral_Traits".

Here is an example format:
{
  "persona": "The Brand Conservative/Practical Buyer",
  "description": "Focuses on finding the best value for their money. Prioritizes safety, reliability, and fuel efficiency over brand appeal or style. Likely to research extensively before making a purchase decision.",
  "keywords": ["Quality", "Safety Features", "Fuel Efficiency", "Practicality", "Reliability", "Affordability", "Low Maintenance"],
  "visual_textual_contents": [
    "Clear and concise information about fuel efficiency and safety ratings",
    "Easy-to-use comparison tools",
    "Simple and straightforward navigation",
    "Focus on value and affordability",
    "Images and videos that showcase the car’s fuel efficiency, safety features, and practicality",
    "Shots of the car in a city setting or on a long road trip",
    "Videos highlighting the car’s handling, braking, and acceleration"
  ],
  "points_of_focus": ["Reviews and comparison", "Safety and Reliability", "Fuel efficiency", "Practical Features"],
  "emotional_drivers": "Make me feel in control",
  "demographics": "Age: 35-55, Income: Middle to High, Occupation: Professional",
  "behavioral_traits": "Thorough researcher, Value-oriented, Risk-averse"
}
"""

# Function to generate persona data
def generate_persona_data(user_input):
    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_input)
    ]

    response = client.chat(
        model=model,
        response_format={"type": "json_object"},
        messages=messages,
    )

    return response.choices[0].message.content

# Example user input
user_input = """
Survey Outcome: Snehal - 27 Years Female, IT Professional

Mindset / Life Values:
- Me + Family (protecting self-identity)
- Independence & Freedom
- Caring - Individuals as well as society
- Open-minded

Emotional Needs:
- Relieve Driving Stress & Anxiety in chaotic city roads
- Exterior design must complement her cheerful lifestyle
- Not just for commute
- Feel youthful
- Feel safe & cared (Self & Family)

Functional Needs:
- Energetic & Cheerful Exterior Design
- Boost driving confidence
- Compact, Easy to drive
- Prevent careless driving mistakes
- Make me relaxed while driving

Unmet Needs:
- Aspires unlimited Freedom but limited by Anxieties (Anxiety in Driving)

Personal Details:
- Gender, Age: Female, 27 years
- Living place: Pune
- Household members: 7 (Father, mother, brother's family with 2 kids)
- Occupation: Sr IT Engineer in MasterCard
- Currently owned vehicle: Hyundai Verna, replaced Hyundai i20 because it was too big
- Type of buyer: Additional Buyer
- Years of ownership: 3 years
"""

# Generate the persona data
persona_data = generate_persona_data(user_input)
print(persona_data)
