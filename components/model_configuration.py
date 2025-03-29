import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()
def model_config():
    GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
    return model