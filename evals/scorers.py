"""
Dual LLM Scorer for Braintrust
Judges whether both Gemini and Groq responses are correct using OpenAI.
"""

import base64
from typing import Any, Dict, Literal

import braintrust
from autoevals import LLMClassifier
from pydantic import BaseModel, Field

from evals.client import get_client

PROJECT_NAME = "Nlp_Writer"


class Input(BaseModel):
    input: Any
    output: Any
    expected: Any
    metadata: Dict[str, Any]


class JudgeResponse(BaseModel):
    """Structured output format for the judge's decision."""

    decision: Literal["YES", "NO"] = Field(
        description="YES if both responses are correct, NO if one or both are incorrect"
    )
    reasoning: str = Field(
        description="Explanation of why both responses are correct or incorrect"
    )


# Base template with placeholder markers
_DUAL_LLM_JUDGE_TEMPLATE = """
You are an expert evaluator tasked with comparing responses from two AI services to determine if both answers are **subjectively correct** for the given question.

## DATA ##

Question:
<question>
__INPUT__
</question>

Gemini Response:
<gemini_response>
__GEMINI_RESPONSE__
</gemini_response>

Meta Response:
<meta_response>
__GROQ_RESPONSE__
</meta_response>

## EVALUATION CRITERIA ##

Your task is to determine if BOTH services provided correct answers.
Focus on factual accuracy, not style, length, or completeness.

### Correctness Definition:
- An answer is CORRECT if it contains the accurate factual information that answers the question
- An answer is CORRECT even if it includes additional information beyond what was asked
- An answer is INCORRECT if it contains factually wrong information or contradicts known facts
- Minor variations in phrasing or formatting do NOT affect correctness

### Decision Rules:

1. **BOTH CORRECT (YES)**
   - Both LLM Gateway and Content Generation Service provide factually accurate answers
   - Additional context or information is acceptable
   - Different phrasings of the same correct fact are acceptable
   
   Examples:
   - Q: "What is the capital of France?"
     - LLM Gateway: "Paris"
     - Content Generation Service: "Paris the capital of France"
     - Result: YES
   - LLM Gateway: "France capital is Paris"
     - Content Generation Service: "Paris the capital of France"
     - Result: YES

2. **ONE OR BOTH INCORRECT (NO)**
   - Either LLM Gateway or Content Generation Service provides factually wrong information
   - Must specify which service(s) provided incorrect information
   
   Example:
   - LLM Gateway: "France capital is London"
     - Content Generation Service: "Paris the capital of France"
     - Result: NO - LLM Gateway answer is incorrect, Content Generation Service answer is correct

3. **BOTH CORRECT WITH NOTES**
   - Both answers are factually correct
   - But one service provides excessive or potentially unhelpful additional information
   - Still counts as YES, but note the verbosity
   
   Example:
   - LLM Gateway: "France capital is London"
     - Content Generation Service: "Paris the capital of France. Paris is beautiful, it has 2 million population"
     - Result: NO - LLM Gateway is incorrect. (Note: If LLM Gateway had been correct, this would be YES with a note about Content Generation Service's verbosity)

Provide your evaluation with:
- decision: "YES" if both responses are correct, "NO" if one or both are incorrect
- reasoning: Your explanation (1-3 sentences)
"""


def get_dual_llm_prompt_for_python() -> str:
    """
    Get the prompt template for Python function scorers (using .format()).
    Replaces placeholders with Python format strings like {input}.
    """
    return (
        _DUAL_LLM_JUDGE_TEMPLATE.replace("__INPUT__", "{input}")
        .replace("__GEMINI_RESPONSE__", "{gemini_response}")
        .replace("__GROQ_RESPONSE__", "{groq_response}")
    )


def get_dual_llm_prompt_for_braintrust() -> str:
    """
    Get the prompt template for Braintrust LLM-as-a-judge scorers (using Mustache).
    Replaces placeholders with Mustache template syntax like {{input}}.
    """
    return (
        _DUAL_LLM_JUDGE_TEMPLATE.replace("__INPUT__", "{{input}}")
        .replace("__GEMINI_RESPONSE__", "{{output.gemini_response}}")
        .replace("__GROQ_RESPONSE__", "{{output.groq_response}}")
    )


# For backward compatibility and ease of use in the function scorer
CODE_BASED_PROMPT = get_dual_llm_prompt_for_python()
CLASSIFIER_PROMPT = get_dual_llm_prompt_for_braintrust()


# Convenience function for easy import
def dual_llm_scorer(
    input: str,
    output: Dict[str, str],
    expected: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Scorer that judges if both Gemini and Groq responses are correct.
    """
    name = "DualLLMScorer"
    if not output:
        return {
            "name": name,
            "score": 0.0,
            "metadata": {
                "error": "No output provided to scorer",
            },
        }

    # Extract responses from task output
    gemini_response = output.get("gemini_response", "")
    groq_response = output.get("groq_response", "")

    if not gemini_response or not groq_response:
        return {
            "name": "DualLLMScorer",
            "score": 0.0,
            "metadata": {
                "error": "Output missing gemini_response or groq_response",
                "output": output,
            },
        }

    client = get_client()

    # Use structured output with Pydantic model
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",  # Structured output requires this model or newer
        messages=[
            {
                "role": "user",
                "content": CODE_BASED_PROMPT.format(
                    input=input,
                    gemini_response=gemini_response,
                    groq_response=groq_response,
                ),
            }
        ],
        response_format=JudgeResponse,
        temperature=0,
    )

    # Extract structured output
    judgment = completion.choices[0].message.parsed

    if judgment is None:
        return {
            "name": "DualLLMScorer",
            "score": 0.0,
            "metadata": {
                "error": "Failed to parse structured output from judge",
                "gemini_response": gemini_response,
                "groq_response": groq_response,
            },
        }

    # Convert decision to score
    score = 1.0 if judgment.decision == "YES" else 0.0

    return {
        "name": "DualLLMScorer",
        "score": score,
        "metadata": {
            "gemini_response": gemini_response,
            "groq_response": groq_response,
            "judge_decision": judgment.decision,
            "judge_reasoning": judgment.reasoning,
            "expected": expected,
        },
    }


def response_time_scorer(
    input: Any,
    output: Any,
    expected: Any,
    metadata: Dict[str, Any],
):
    DEFAULT_RESPONSE_TIME_THRESHOLD = 10
    NAME = "ResponseTimeScorer"

    gemini_response_time = output.get("gemini_duration_seconds", None)
    if gemini_response_time is None:
        return {
            "name": NAME,
            "score": 0.0,
            "metadata": {
                "error": "Gemini response time is not provided",
                "output": output,
                **metadata,
            },
        }

    if metadata and metadata.get("Expected Latency in Seconds", None) is not None:
        DEFAULT_RESPONSE_TIME_THRESHOLD = metadata["Expected Latency in Seconds"]

    score = 1.0 if gemini_response_time < DEFAULT_RESPONSE_TIME_THRESHOLD else 0.0
    return {
        "name": NAME,
        "score": score,
        "metadata": {
            "gemini_response_time": gemini_response_time,
            "response_time_threshold": DEFAULT_RESPONSE_TIME_THRESHOLD,
            **metadata,
        },
    }


dual_llm_scorer_classifier = LLMClassifier(
    name="Dual LLM Scorer Classifier",
    prompt_template=CLASSIFIER_PROMPT,
    choice_scores={
        "YES": 1.0,
        "NO": 0.0,
    },
    use_cot=True,
    model="gpt-4o-2024-08-06",
)


def dual_llm_scorer_classifier_images(
    input: str,
    output: Dict[str, Any],
    expected: str = "",
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Scorer for evaluating dual LLM responses with image context.

    Expects:
    - metadata["attachment"]: URL or base64 data URI of the image
    - output["gemini_response"]: Response from Gemini
    - output["groq_response"]: Response from Groq
    """
    name = "DualLLMScorerImages"
    metadata = metadata or {}

    # Extract image URL from metadata
    attachment = metadata.get("attachment")
    if not attachment:
        return {
            "name": name,
            "score": 0.0,
            "metadata": {
                "error": "No image attachment found in metadata",
            },
        }

    # Extract image from metadata
    image_bytes = attachment.data
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    content_type = attachment.reference["content_type"]

    # Extract responses
    gemini_response = output.get("gemini_response", "")
    groq_response = output.get("groq_response", "")

    if not gemini_response or not groq_response:
        return {
            "name": name,
            "score": 0.0,
            "metadata": {
                "error": "Output missing gemini_response or groq_response",
            },
        }

    client = get_client()

    # Build the prompt text
    prompt_text = f"""
You are an expert evaluator assessing the factual correctness of AI responses about an image.

## CONTEXT ##
Question: {input}

Gemini Response: {gemini_response}

Groq Response: {groq_response}

## YOUR TASK ##
Determine if BOTH responses are factually correct based on the visual content of the image.

Focus on:
- Factual accuracy of descriptions
- Correct identification of objects, people, or scenes
- Accurate spatial relationships and details

Return:
- "BOTH_CORRECT" if both responses are accurate
- "ONE_OR_BOTH_INCORRECT" if at least one response has factual errors
"""

    if expected:
        prompt_text += f"\n\nReference Answer (if helpful): {expected}"

    # Structure the message with both text and image
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_text,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{content_type};base64,{image_base64}"},
                },
            ],
        }
    ]

    # Use structured output
    class ImageJudgeResponse(BaseModel):
        decision: Literal["BOTH_CORRECT", "ONE_OR_BOTH_INCORRECT"] = Field(
            description="Whether both responses are factually correct"
        )
        reasoning: str = Field(
            description="Explanation based on visual analysis of the image"
        )

    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=messages,  # type: ignore
            response_format=ImageJudgeResponse,
            temperature=0,
        )

        judgment = completion.choices[0].message.parsed

        if judgment is None:
            return {
                "name": name,
                "score": 0.0,
                "metadata": {
                    "error": "Failed to parse structured output from judge",
                },
            }

        # Convert decision to score
        score = 1.0 if judgment.decision == "BOTH_CORRECT" else 0.0

        return {
            "name": name,
            "score": score,
            "metadata": {
                "gemini_response": gemini_response,
                "groq_response": groq_response,
                "judge_decision": judgment.decision,
                "judge_reasoning": judgment.reasoning,
                "image_url": attachment,
            },
        }
    except Exception as e:
        return {
            "name": name,
            "score": 0.0,
            "metadata": {
                "error": f"Error calling judge: {str(e)}",
                "gemini_response": gemini_response,
                "groq_response": groq_response,
            },
        }


project = braintrust.projects.create(name="Nlp_Writer")

project.scorers.create(
    name="Dual LLM Scorer",
    slug="dual-llm-scorer",
    description="A dual LLM scorer",
    parameters=Input,
    handler=dual_llm_scorer,
)

project.scorers.create(
    name="Response Time Scorer",
    slug="response-time-scorer",
    description="A response time scorer",
    parameters=Input,
    handler=response_time_scorer,
)

project.scorers.create(
    name="Dual LLM Scorer Classifier",
    slug="dual-llm-scorer-classifier",
    description="A dual LLM scorer classifier",
    messages=[
        {"role": "user", "content": CLASSIFIER_PROMPT},
    ],
    model="gpt-4o",
    use_cot=True,
    choice_scores={
        "YES": 1.0,
        "NO": 0.0,
    },
)
