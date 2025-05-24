# test_config.py
from src.config import GOOGLE_API_KEY

if GOOGLE_API_KEY and GOOGLE_API_KEY != "YOUR_GOOGLE_API_KEY_HERE":
    print("SUCCESS: GOOGLE_API_KEY loaded successfully from .env!")
    # print(f"API Key starts with: {GOOGLE_API_KEY[:5]}...") # Optional: to see part of the key
else:
    print("ERROR: GOOGLE_API_KEY not loaded or still set to placeholder.")
    print("Please ensure your .env file is correctly set up with your key.")