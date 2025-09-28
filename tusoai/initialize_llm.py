from openai import OpenAI
import os
	
def init(api_key, openrouter=False):
    if openrouter:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    else:
        os.environ['OPENAI_API_KEY'] = api_key
        client = OpenAI()
    return client
