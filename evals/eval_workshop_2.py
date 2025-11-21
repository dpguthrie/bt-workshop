import time

import braintrust

from evals.client import get_client
from evals.scorers import dual_llm_scorer, response_time_scorer

# Get the Braintrust proxy client
client = get_client()


PROJECT_NAME = "Nlp_Writer"
DATASET_NAME = "Nlp_Generic"


def task(input: str) -> dict:
    """
    Task that calls both Gemini and Groq with the same query.
    Returns both responses for the scorer to judge, along with timing data.
    """

    # Call Gemini and measure time
    gemini_content = ""
    gemini_duration = 0.0
    try:
        gemini_start = time.time()
        gemini_response = client.chat.completions.create(
            model="gpt-4o",  # gemini-2.0-flash-lite
            messages=[{"role": "user", "content": input}],
        )
        gemini_duration = time.time() - gemini_start
        gemini_content = gemini_response.choices[0].message.content or ""  # type: ignore
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
            model="claude-sonnet-4-20250514",  # "llama-3.1-8b-instant
            messages=[{"role": "user", "content": input}],
        )
        groq_duration = time.time() - groq_start
        groq_content = groq_response.choices[0].message.content or ""  # type: ignore
    except Exception as e:
        groq_duration = time.time() - groq_start if "groq_start" in locals() else 0.0
        groq_content = f"ERROR: {str(e)}"
        braintrust.current_span().log(error=f"Groq call failed: {str(e)}")

    # Log custom metrics
    braintrust.current_span().log(
        metrics={
            "gemini_duration_seconds": round(gemini_duration, 3),
            "groq_duration_seconds": round(groq_duration, 3),
        }
    )

    # Return both responses and timing data for the scorer
    # Note: The output dict keys become template variables in the scorer
    return {
        "gemini_response": gemini_content,
        "groq_response": groq_content,
        "gemini_duration_seconds": round(gemini_duration, 3),
        "groq_duration_seconds": round(groq_duration, 3),
    }


#

eval = braintrust.Eval(
    PROJECT_NAME,
    experiment_name="Workshop 2",
    data=braintrust.init_dataset(
        PROJECT_NAME,
        DATASET_NAME,
    ),
    task=task,
    max_concurrency=10,
    scores=[dual_llm_scorer, response_time_scorer],  # type: ignore
)
