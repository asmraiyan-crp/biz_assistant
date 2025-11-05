import os
import json
import warnings
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from datetime import datetime

# Load the .env file
load_dotenv()

try:
    import google.generativeai as genai

    # Configure the client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY not found in .env file.")
        USE_GEMINI = False
        model = None
    else:
        genai.configure(api_key=api_key)

        # --- Model for Summary ---
        generation_config_summary = {
            "temperature": 0.7,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2048,
        }

        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        model_summary = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config=generation_config_summary,
            safety_settings=safety_settings
        )

        # --- Model for Chat ---
        generation_config_chat = {
            "temperature": 0.5,  # Slightly more deterministic for chat
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 500,  # Shorter answers for chat
        }

        model_chat = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config=generation_config_chat,
            safety_settings=safety_settings
        )

        USE_GEMINI = True

except Exception as e:
    print(f"Warning: Failed to initialize Google Gemini: {e}")
    USE_GEMINI = False
    model_summary = None
    model_chat = None


# --- Custom JSON Encoder (no changes) ---
class CustomEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle special types like Timestamps and numpy types.
    """

    def default(self, obj):
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        if isinstance(obj, (np.int64, np.int32, np.int16)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(CustomEncoder, self).default(obj)


# --- Summary Function (no changes) ---
def llm_summary(insights, evidence=[]):
    """
    Generates a board-level summary and recommendations.
    """
    if USE_GEMINI and model_summary:
        prompt = """You are an analyst. Given the numeric insights (JSON) and evidence snippets, write a concise board-level summary (3-6 sentences) and 2 action recommendations."""

        try:
            insights_json = json.dumps(insights, indent=2, cls=CustomEncoder)
        except Exception as e:
            print(f"Error serializing insights: {e}")
            insights_json = json.dumps({"error": "Failed to serialize insights"})

        prompt += "\n\nINSIGHTS:\n" + insights_json

        if evidence:
            prompt += "\n\nEVIDENCE SNIPPETS:\n" + "\n".join(evidence[:5])

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                resp = model_summary.generate_content(prompt)

            if resp.parts:
                return resp.parts[0].text.strip()
            else:
                finish_reason = resp.candidates[0].finish_reason if (
                            resp.candidates and resp.candidates[0].finish_reason) else 'UNKNOWN'
                return f"[LLM generation failed] No text part returned from API. Finish Reason: {finish_reason}"

        except Exception as e:
            return f'[LLM generation failed] Error during generation: {str(e)}'

    # Fallback logic
    print("Falling back to non-LLM summary.")
    lines = []
    for p in insights.get('products', []):
        lines.append(
            f"{p['product_name']}: forecast mean {p.get('forecast_mean', 0):.1f}, risk_index {p.get('risk_index', 0.0):.2f}, P(stockout) {p.get('p_stockout', 0.0):.2f}.")
    recs = ["Check high-risk products and consider reordering.",
            "Investigate negative reviews for products with low sentiment."]
    return '\n'.join(lines + recs)


# --- *** NEW CHATBOT FUNCTION *** ---
def llm_chat(query: str, context_snippets: list[str]) -> str:
    """
    Answers a user's query based on provided context snippets.
    This is the core of the RAG (Retrieval-Augmented Generation) chatbot.
    """
    if not USE_GEMINI or not model_chat:
        return "I'm sorry, the AI chat service is not available right now."

    if not context_snippets:
        # If the RAG search found no relevant documents
        return "I'm sorry, I couldn't find any relevant reviews to answer that question. Please try rephrasing."

    # Build the prompt
    prompt = f"You are a helpful AI assistant. Answer the user's question based *only* on the provided context snippets. If the answer isn't in the context, say so.\n\n"
    prompt += "CONTEXT SNIPPETS:\n"
    for i, snippet in enumerate(context_snippets):
        prompt += f"{i + 1}. \"{snippet}\"\n"

    prompt += f"\nUSER QUESTION:\n{query}\n\nANSWER:\n"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resp = model_chat.generate_content(prompt)

        if resp.parts:
            return resp.parts[0].text.strip()
        else:
            finish_reason = resp.candidates[0].finish_reason if (
                        resp.candidates and resp.candidates[0].finish_reason) else 'UNKNOWN'
            return f"[AI chat failed] No text part returned. Finish Reason: {finish_reason}"

    except Exception as e:
        return f'[AI chat failed] Error: {str(e)}'