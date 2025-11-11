import os

from braintrust.oai import wrap_openai
from openai import OpenAI


def get_client():
    return wrap_openai(
        OpenAI(
            base_url="https://api.braintrust.dev/v1/proxy",
            api_key=os.environ["BRAINTRUST_API_KEY"],
        )
    )
