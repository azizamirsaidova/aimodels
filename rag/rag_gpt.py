import openai
from rag_model import retreive_and_agument
openai.api_key = OPENAI_API_TOKEN

prompts = ['what pasta dishes do you have', 'what events do you guys do', 'oh cool what is karaoke']

for prompt in prompts:

    ra_prompt = retreive_and_agument(prompt)
    response = openai.Completion.create(model="gpt-3.5-turbo-instruct", prompt=ra_prompt, max_tokens=80).choices[0].text

    print(f'prompt: "{prompt}"')
    print(f'response: {response}')