"""
utils/gemini_processor.py

Sends the complete meeting transcript to the Google Gemini API
(`gemini-1.5-flash`) and returns a structured Minutes of Meeting (MoM)
object containing: summary, key decisions, and a list of actionable tasks.
"""

import os
import json
import logging
import re

import google.generativeai as genai
from dotenv import load_dotenv
from google.generativeai.types import HarmCategory, HarmBlockThreshold

load_dotenv()

logger = logging.getLogger(__name__)

# ── System Prompt ─────────────────────────────────────────────────────────────
# This prompt is carefully engineered to produce consistent, parseable JSON.
SYSTEM_PROMPT = """You are an expert NGO meeting secretary. You will be given a raw transcript of an NGO meeting.

Your job is to produce a structured Minutes of Meeting (MoM) document.

You MUST respond with ONLY a valid JSON object — no markdown fences, no preamble, no explanation outside the JSON.

The JSON structure must be exactly:

{
  "meeting_title_suggestion": "<short inferred title for this meeting>",
  "summary": "<2-4 sentence executive summary of the entire meeting>",
  "key_decisions": [
    "<decision 1>",
    "<decision 2>"
  ],
  "discussion_points": [
    {
      "topic": "<topic heading>",
      "notes": "<brief notes on what was discussed>"
    }
  ],
  "tasks": [
    {
      "description": "<clear, actionable task description>",
      "assignee": "<person's name or 'Unassigned' if not mentioned>",
      "due_date": "<mentioned due date or null>",
      "priority": "<High | Medium | Low>"
    }
  ],
  "next_meeting_date": "<mentioned date or null>"
}

Rules:
- Extract EVERY distinct action item or task mentioned.
- If a person is mentioned alongside a task, set them as assignee.
- If no assignee is clear, use "Unassigned".
- Key decisions are things that were agreed upon, approved, or resolved.
- Be factual and concise. Do NOT add information not present in the transcript.
- Respond with ONLY the JSON object."""


def _clean_json_response(text: str) -> str:
    """Strip markdown code fences if the model accidentally wraps output."""
    # Remove ```json ... ``` or ``` ... ``` wrappers
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    return text.strip()


def generate_mom(transcript: str) -> dict:
    """
    Process a meeting transcript through Gemini and return the structured MoM.

    Args:
        transcript: The full text transcript of the meeting.

    Returns:
        A dict matching the JSON schema above.

    Raises:
        ValueError: If the API returns unparseable JSON.
        google.generativeai.types.GenerationException: On API failure.
    """
    if not transcript or len(transcript.strip()) < 50:
        raise ValueError(
            "Transcript is too short to generate a meaningful MoM. "
            "Please check the audio file and transcription."
        )

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash", # Consider changing to "gemini-1.5-pro" if flash keeps failing
        system_instruction=SYSTEM_PROMPT,
        safety_settings=safety_settings, # Applied here!
        generation_config=genai.GenerationConfig(
            temperature=0.2, 
            max_output_tokens=8192, # Bumped to 8k just in case
            response_mime_type="application/json",
        ),
    )

    # Build user message
    user_message = (
        f"Please process the following meeting transcript and return the MoM JSON:\n\n"
        f"--- TRANSCRIPT START ---\n{transcript}\n--- TRANSCRIPT END ---"
    )

    logger.info(
        "Sending transcript to Gemini (%d characters)...", len(transcript)
    )

    response = model.generate_content(user_message)
    raw_text = response.text

    logger.info("Gemini response received (%d characters)", len(raw_text))

    # Parse JSON
    clean_text = _clean_json_response(raw_text)
    try:
        mom_data = json.loads(clean_text)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse Gemini JSON response: %s\nRaw: %s", e, clean_text[:500])
        raise ValueError(
            f"Gemini returned invalid JSON. Parse error: {e}. "
            "Check logs for raw response."
        ) from e

    # Validate required fields with sensible defaults
    mom_data.setdefault("meeting_title_suggestion", "Untitled Meeting")
    mom_data.setdefault("summary", "No summary available.")
    mom_data.setdefault("key_decisions", [])
    mom_data.setdefault("discussion_points", [])
    mom_data.setdefault("tasks", [])
    mom_data.setdefault("next_meeting_date", None)

    logger.info(
        "MoM generated: %d tasks, %d decisions, %d discussion points",
        len(mom_data["tasks"]),
        len(mom_data["key_decisions"]),
        len(mom_data["discussion_points"]),
    )

    return mom_data
