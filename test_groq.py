import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

API_KEY = os.environ.get("GROQ_API_KEY")

if not API_KEY:
    print("Failed to load GROQ_API_KEY from environment or .env file.")
else:
    print(f"Loaded API Key: {API_KEY[:4]}...{API_KEY[-4:]}")
    try:
        client = Groq(api_key=API_KEY)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "Respond with the word 'Success!' if you can read this."
                }
            ],
            model="groq/compound",
            temperature=0,
            max_tokens=10
        )
        print("API Response:", chat_completion.choices[0].message.content)
    except Exception as e:
        print("API Error:", e)
