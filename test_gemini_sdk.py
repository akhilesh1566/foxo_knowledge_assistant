# test_gemini_sdk.py
import google.generativeai as genai
from src.config import GOOGLE_API_KEY

try:
    if not GOOGLE_API_KEY:
        print("GOOGLE_API_KEY not found. Check src/config.py and .env file.")
    else:
        genai.configure(api_key=GOOGLE_API_KEY)

        print("Attempting to list models...")
        # List available models (a simple, non-quota-intensive call)
        model_count = 0
        for m in genai.list_models():
            # Typically, you'd check if 'generateContent' is supported for a specific model.
            # For this test, just listing them is enough.
            # print(f"Model: {m.name}")
            model_count += 1

        if model_count > 0:
            print(f"SUCCESS: Successfully listed {model_count} models using Google Generative AI SDK.")
        else:
            print("WARNING: Listed 0 models. This might be okay if your API key has restrictions, or it might indicate an issue.")

        # Optional: Try a very simple generation (uses quota)
        # print("\nAttempting a simple text generation with gemini-pro...")
        # model = genai.GenerativeModel('gemini-pro')
        # response = model.generate_content("Tell me a fun fact about Python programming language.", request_options={'timeout': 60})
        # print("Response from Gemini Pro:")
        # print(response.text)
        # print("SUCCESS: Simple text generation successful.")

except Exception as e:
    print(f"ERROR during Google Generative AI SDK test: {e}")
    print("Things to check:")
    print("1. Is your GOOGLE_API_KEY correct and active?")
    print("2. Do you have internet connectivity?")
    print("3. Have you enabled the 'Generative Language API' (or similar) in your Google Cloud Project if using a project key?")
    print("   (For keys from AI Studio, this is usually pre-configured).")
