import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROJECT_NAME = os.environ["PROJECT_NAME"]


def get_client():
    return OpenAI(
        base_url="https://api.braintrust.dev/v1/proxy",
        api_key=os.environ["BRAINTRUST_API_KEY"],
        default_headers={
            "x-bt-parent": f"project_name:{PROJECT_NAME}",
            "x-bt-org-name": "Braintrust Demos",
            "x-bt-use-cache": "never",
        },
    )
