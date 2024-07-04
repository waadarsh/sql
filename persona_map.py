from dotenv import load_dotenv
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# Set your Mistral AI API key
load_dotenv('.env')

# Define the model
model = "mistral-small-latest"

# Initialize the Mistral Client
client = MistralClient()

# The second system prompt to define the task of mapping the generated persona to existing personas

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

Here are additional example personas:
{
  "persona": "The Status Chasers/Status Seekers",
  "description": "Values prestige, luxury, and status in their automobile choice. Attracted to high-end brands, exquisite craftsmanship, advanced technology, and luxurious interiors. Motivated by the desire to impress others and showcase their wealth or social status.",
  "keywords": ["Luxury", "Exclusivity", "Sports car", "Premium Features", "High Performance", "Prestige", "Limited Editions"],
  "visual_textual_contents": [
    "High-end driving experience",
    "Textual content focused on luxury features like advanced technology, high-quality materials, and high-performance engines",
    "Highlighting the prestige and exclusivity associated with owning a luxury car",
    "High-quality images and videos showcasing the car’s design and luxury features",
    "Sleek and modern interface reflecting the brand’s image",
    "Customization options to add premium features",
    "Focus on exclusivity and prestige"
  ],
  "points_of_focus": ["Luxury Features", "Brand Prestige", "Performance and power", "Design and style", "Exclusivity"],
  "emotional_drivers": "Make me feel special",
  "demographics": "Age: 30-50, Income: High, Occupation: Executive",
  "behavioral_traits": "Image-conscious, Trendsetter, Brand-loyal"
},
{
  "persona": "The Adventure Seeker",
  "description": "Enjoys outdoor activities, off-roading, or exploring rugged terrains. Interested in vehicles with off-road capabilities, durable build quality, and features like all-wheel drive or robust suspension systems.",
  "keywords": ["Off Road", "Adventure", "Long Drive", "Never tired", "Power", "Ready to Go"],
  "visual_textual_contents": [
    "Prominent display of off-road capabilities and adventure-ready features",
    "Clear information about towing capacity and durability",
    "Rugged and outdoorsy interface reflecting the vehicle’s capabilities",
    "Emphasis on reliability and toughness",
    "Images and videos highlighting the car’s off-road capabilities",
    "Shots of the car in rugged terrain or towing heavy loads",
    "Videos showcasing the car’s durability and toughness, such as extreme weather conditions"
  ],
  "points_of_focus": ["Off Road Capability", "Durability", "Towing Capability", "All wheel drive", "Adventure ready Features", "Safety and performance"],
  "emotional_drivers": "Make me feel the power/rush",
  "demographics": "Age: 25-45, Income: Middle to High, Occupation: Adventurer",
  "behavioral_traits": "Thrill-seeker, Nature-lover, Active lifestyle"
},
{
  "persona": "The Contentment Settler",
  "description": "Prioritizes affordability, fuel efficiency, and low maintenance costs. Interested in reliable and economical vehicles with good resale value, affordable insurance rates, and high fuel efficiency.",
  "keywords": ["Stability", "Satisfaction", "Safety and Security", "Risk Aversion", "Comfort and Convenience", "Personal Satisfaction", "Decision-Making Factors", "Balanced", "Emotional Connection"],
  "visual_textual_contents": [
    "High-end driving experience",
    "Focus on affordability, insurance rates, and fuel efficiency",
    "High-quality images and videos showcasing the car’s design and luxury features",
    "Sleek and modern interface reflecting the brand’s image",
    "Customization options to add premium features",
    "Focus on exclusivity and prestige"
  ],
  "points_of_focus": ["Safety", "Personalisation Options", "Easy Maintenance", "Brand loyalty", "Aesthetic Appeal", "Space and comfort", "Customer service", "Brand reputation", "Easy to use", "Test Drive"],
  "emotional_drivers": "Make me feel cared for",
  "demographics": "Age: 40-60, Income: Middle, Occupation: Teacher",
  "behavioral_traits": "Practical, Stability-oriented, Careful planner"
},
{
  "persona": "The Seamless Seeker",
  "description": "Prioritizes convenience, efficiency, and a seamless experience in their purchasing decisions.",
  "keywords": ["Efficiency", "Convenience", "Seamless integration", "Quality", "Comfort", "Time saving", "Minimal Friction", "Smart Features"],
  "visual_textual_contents": [
    "Highlight tech-focused features and streamlined buying process",
    "Transparent pricing information",
    "Customer reviews and testimonials",
    "Clean and minimalistic design",
    "Showcase seamless integration of technology within the car",
    "Features like touchscreen displays, voice control, and smartphone connectivity",
    "Interactive and user-friendly interfaces",
    "Time-saving features",
    "Efficient performance: showcase the car’s performance such as hybrid or electric powertrains, aerodynamic design, or fuel-saving technologies"
  ],
  "points_of_focus": ["Safety", "Versatility", "Reliability", "Aesthetic Appeal", "Space and comfort", "Efficiency", "Entertainment and connectivity", "Price and affordability", "Easy to use", "Test Drive"],
  "emotional_drivers": "Make me feel excited",
  "demographics": "Age: 25-40, Income: Middle to High, Occupation: IT Professional",
  "behavioral_traits": "Tech-savvy, Efficiency-driven, Early adopter"
},
{
  "persona": "The Moral Compass/Eco-Conscious/Sustainability Fan",
  "description": "Prioritizes environmental sustainability and fuel efficiency. Interested in hybrid or electric vehicles that offer reduced emissions and improved fuel economy.",
  "keywords": ["Sustainability", "Hybrid", "Emissions", "Social Awareness", "Comfort and Convenience", "Eco Friendly", "Fuel efficiency", "Charging Infrastructure", "Emotional Connection"],
  "visual_textual_contents": [
    "Effectively communicate the eco-friendly and ethical aspects of the car",
    "Showcase environmentally friendly features like hybrid or electric drivetrains, solar panels, regenerative braking, or eco-friendly materials",
    "Feature imagery of the car in natural settings",
    "Certifications and eco-labels",
    "Images and videos highlighting the car’s eco-friendliness",
    "Shots of the car charging or driving in natural settings",
    "Videos demonstrating the car’s efficiency and low emissions"
  ],
  "points_of_focus": ["Sustainability", "Minimalism", "Electric Capabilities", "Charging Infrastructure", "Hybrid", "Safety and performance"],
  "emotional_drivers": "Make me feel cared for",
  "demographics": "Age: 30-50, Income: Middle to High, Occupation: Environmentalist",
  "behavioral_traits": "Eco-conscious, Community-focused, Ethical consumer"
},
{
  "persona": "The Family-Oriented",
  "description": "Includes individuals or couples with children who prioritize safety, ample space, and practicality. Interested in features like spacious interiors, advanced safety systems, and versatile seating arrangements.",
  "keywords": ["Spaciousness", "Safety", "Feel Connected", "Versatility", "Comfort", "Child friendly features", "Balanced", "Reliability"],
  "visual_textual_contents": [
    "Focus on spacious interiors and advanced safety systems",
    "High-quality images and videos showcasing the car’s design and luxury features",
    "Sleek and modern interface reflecting the brand’s image",
    "Customization options to add premium features",
    "Focus on exclusivity and prestige"
  ],
  "points_of_focus": ["Safety", "Versatility", "Reliability", "Aesthetic Appeal", "Space and comfort", "Efficiency", "Entertainment and connectivity", "Price and affordability", "Easy to use", "Test Drive"],
  "emotional_drivers": "Make me feel safe and secure",
  "demographics": "Age: 30-45, Income: Middle, Occupation: Parent",
  "behavioral_traits": "Family-focused, Safety-conscious, Practical"
}

When you receive a new persona, map it to the existing personas or create a mix and match or even a subset of the existing personas.
return the mapped persona as a json object.
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

user_input = """
{
  "persona": "The Youthful & Anxious Commuter",
  "description": "A young, independent IT professional who values her family and her sense of identity. She seeks a vehicle that reduces her driving stress and anxiety, provides a sense of safety for herself and her loved ones, and complements her energetic and cheerful lifestyle.",
  "keywords": [
    "Anxiety Reduction",
    "Compact Size",
    "Energetic Design",
    "Driving Confidence",
    "Safety Features"
  ],
  "visual_textual_contents": [
    "Images and videos showcasing the car's compact size and easy maneuverability",
    "Photos of the cheerful exterior design",
    "Videos demonstrating advanced safety features and driver assistance technologies",
    "Testimonials from other anxious drivers who have found relief with the vehicle"
  ],
  "points_of_focus": [
    "Anxiety-reducing features",
    "Compact size and ease of driving",
    "Safety features",
    "Cheerful exterior design"
  ],
  "emotional_drivers": "Make me feel safe, confident, and relaxed while driving",
  "demographics": "Gender: Female, Age: 27, Occupation: Sr IT Engineer, Income: Middle to High",
  "behavioral_traits": "Anxious driver, Values safety, Seeks independence & freedom, Family-oriented"
}
"""

# Function to generate persona data
def map_persona_data(user_input):
    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_input)
    ]

    response = client.chat(
        model=model,
        response_format={"type": "json_object"},
        messages=messages,
        temperature=0,
    )

    return response.choices[0].message.content

# Generate the mapped data
persona_data = map_persona_data(user_input)
print(persona_data)
