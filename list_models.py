import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq()

models = client.models.list()
for m in models.data:
    print(m.id)
