import openai

# Set up OpenAI API key
openai.api_key = 'your-api-key'

def generate_sdxl_prompt(car_data, user_persona):
    # Define the system prompt with placeholders for car data and user persona
    system_prompt = f"""
    You are an AI assistant that generates detailed SDXL image generation prompts. 
    Given the following car data and user persona, create a detailed and visually rich prompt for generating an image.

    Car Data:
    {car_data}

    User Persona:
    {user_persona}

    Create an SDXL prompt based on the above information.
    """
    
    # Define the messages for the chat completion
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Generate the SDXL image prompt."}
    ]
    
    # Call the OpenAI ChatCompletion API
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=200
    )
    
    # Extract and return the generated prompt
    sdxl_prompt = response['choices'][0]['message']['content'].strip()
    return sdxl_prompt

# Example usage
car_data = """
Model: Tesla Model S
Color: Midnight Silver Metallic
Features: Autopilot, Panoramic Sunroof, 21-inch Wheels
"""
user_persona = """
Name: Alex
Age: 35
Occupation: Software Engineer
Hobbies: Tech Gadgets, Eco-friendly Living, Modern Design
"""
sdxl_prompt = generate_sdxl_prompt(car_data, user_persona)
print(sdxl_prompt)
