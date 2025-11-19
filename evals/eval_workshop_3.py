import base64
import time

import braintrust
from dotenv import load_dotenv

from evals.client import get_client
from evals.scorers import dual_llm_scorer_classifier_images

load_dotenv()

# Get the Braintrust proxy client
client = get_client()


PROJECT_NAME = "Nlp_Writer"
DATASET_NAME = "Nlp_Generic_Images"


def task(input: str, hooks: braintrust.EvalHooks) -> dict:
    """
        Task that calls both Gemini and Groq with the same query + picture from
    metadata.
        Returns both responses for the scorer to judge, along with timing data.
    """

    # Get picture from metadata
    picture_bytes = hooks.metadata["attachment"].data
    picture_base64 = base64.b64encode(picture_bytes).decode("utf-8")
    content_type = hooks.metadata["attachment"].reference["content_type"]

    # Build message with text + image
    message_content = [
        {"type": "text", "text": input},
        {
            "type": "image_url",
            "image_url": {"url": f"data:{content_type};base64,{picture_base64}"},
        },
    ]

    # Call Gemini and measure time
    gemini_content = ""
    gemini_duration = 0.0
    try:
        gemini_start = time.time()
        gemini_response = client.chat.completions.create(
            model="gemini-2.0-flash-lite",
            messages=[{"role": "user", "content": message_content}],
        )
        gemini_duration = time.time() - gemini_start
        gemini_content = gemini_response.choices[0].message.content or ""
    except Exception as e:
        gemini_duration = (
            time.time() - gemini_start if "gemini_start" in locals() else 0.0
        )
        gemini_content = f"ERROR: {str(e)}"
        braintrust.current_span().log(error=f"Gemini call failed: {str(e)}")

    # Call Groq and measure time
    groq_content = ""
    groq_duration = 0.0
    try:
        groq_start = time.time()
        groq_response = client.chat.completions.create(
            model="claude-sonnet-4-20250514",  # "meta-llama/llama-4-maverick-17b-128e-instruct
            messages=[{"role": "user", "content": message_content}],
        )
        groq_duration = time.time() - groq_start
        groq_content = groq_response.choices[0].message.content or ""
    except Exception as e:
        groq_duration = time.time() - groq_start if "groq_start" in locals() else 0.0
        groq_content = f"ERROR: {str(e)}"
        braintrust.current_span().log(error=f"Claude call failed: {str(e)}")

    # Log custom metrics
    braintrust.current_span().log(
        metrics={
            "gemini_duration_seconds": round(gemini_duration, 3),
            "groq_duration_seconds": round(groq_duration, 3),
        }
    )

    # Return both responses and timing data for the scorer
    return {
        "gemini_response": gemini_content,
        "groq_response": groq_content,
        "gemini_duration_seconds": round(gemini_duration, 3),
        "groq_duration_seconds": round(groq_duration, 3),
    }


eval = braintrust.Eval(
    PROJECT_NAME,
    experiment_name="Workshop 3",
    data=braintrust.init_dataset(
        PROJECT_NAME,
        DATASET_NAME,
    ),
    task=task,
    scores=[  # type: ignore
        dual_llm_scorer_classifier_images,  # type: ignore
    ],
    max_concurrency=10,
)
