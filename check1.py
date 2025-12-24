import google.generativeai as genai
from PIL import Image
import os

# Configure the API key (it will automatically pick up from the environment variable)
# Alternatively, you can pass it directly: genai.configure(api_key="YOUR_API_KEY")
genai.configure(api_key="AIzaSyCAs7Ltw75a6KWaudV8seVvuE3hwDG-e9k")

# 1. Load the model (use a multimodal model like 'gemini-2.5-flash' or 'gemini-2.0-pro')
model = genai.GenerativeModel('gemini-2.5-flash')

# 2. Open the image
try:
    # Replace 'example.jpg' with the path to your image file
    img = Image.open('C:\\Users\\cyril\\Documents\\GitHub\\infolks-projects\\R&D\\RAG Bot V2\\pexels-magda-ehlers-pexels-1340185.jpg')
except FileNotFoundError:
    print("Error: example.jpg not found. Please provide a valid image path.")
    exit()

# 3. Create a prompt for the AI
prompt = "Generate a detailed caption for this image."

# 4. Generate content by passing both the prompt and the image
response = model.generate_content([prompt, img])

# 5. Print the generated caption
print(f"Generated Caption:\n{response.text}")
