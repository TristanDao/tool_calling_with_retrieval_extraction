import os

from dotenv import load_dotenv
from openai import OpenAI


# load_dotenv()

client = OpenAI(
    api_key="sk-xdDJXNEGs1obbi3zPTzJqpZOuWL7u5zWwodRnDJw4SItOPipB101Zalbv4lZpntL",
    base_url="https://opencode.ai/zen/go/v1",
)
completion = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": "Nhắc tên tôi 3 lần"}],
)
print(completion.choices[0].message.content)

