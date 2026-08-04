import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(
    api_key=os.environ["ALIBABA_API_KEY"],
    base_url=os.environ["ALIBABA_URL"],
)
completion = client.chat.completions.create(
    model=os.getenv("ALIBABA_MODEL", "qwen-mt-flash"),
    messages=[{"role": "user", "content": "Who are you?"}],
)
print(completion.choices[0].message.content)
