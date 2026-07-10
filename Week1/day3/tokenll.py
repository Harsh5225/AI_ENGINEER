import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables.")

client=Groq(api_key=my_api_key) 
model="llama-3.3-70b-versatile"
role="user"
prompt1= "Hi!"
prompt2="Explain time travel in Detail."
prompt3="Write an essay on the beauty of nature."

prompts=[prompt1, prompt2, prompt3]
for prompt in prompts:
    message={
        "role": role,
        "content": prompt
    }
    messages=[message]
    response=client.chat.completions.create(model=model, messages=messages)
    usage=response.usage
    print(f"Prompt: {prompt}--> your token usage is: {usage.prompt_tokens}, completion_tokens:{usage.completion_tokens},total_tokens:{usage.total_tokens}")
    # print(response.choices[0].message.content)

