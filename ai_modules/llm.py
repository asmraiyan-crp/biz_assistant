import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import warnings

try:
    import google.generativeai as genai

    # Configure the client with the API key from the environment variable
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    # Set up the model
    generation_config = {
        "temperature": 0.7,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 300,  # Max tokens for the summary
    }

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    # Use gemini-1.5-flash: It's fast, high-quality, and great for the free tier
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    USE_GEMINI = True
except Exception as e:
    print(f"Warning: Failed to initialize Google Gemini: {e}")
    USE_GEMINI = False


# --- ROBUST JSON ENCODER ---
# This class solves 'Timestamp' and 'numpy' serialization errors
class CustomEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle special types that the
    default json library can't serialize, like Timestamps and numpy types.
    """

    def default(self, obj):
        if isinstance(obj, (datetime, pd.Timestamp)):
            # Convert any datetime or Timestamp object to an ISO 8601 string
            return obj.isoformat()
        if isinstance(obj, (np.int64, np.int32, np.int16)):
            # Convert numpy integers to standard Python int
            return int(obj)
        if isinstance(obj, (np.float64, np.float32, np.float16)):
            # Convert numpy floats to standard Python float
            return float(obj)
        if isinstance(obj, np.ndarray):
            # Convert numpy arrays to standard Python lists
            return obj.tolist()

        # Let the base class handle default cases
        return super(CustomEncoder, self).default(obj)


# --- END ENCODER ---

def llm_summary(insights, evidence=[]):
    """
    Generates a board-level summary and recommendations using Google Gemini
    or a fallback rule-based summary.
    """
    if USE_GEMINI and model:
        prompt = """You are an analyst. Given the numeric insights (JSON) and evidence snippets, write a concise board-level summary (3-6 sentences) and 2 action recommendations."""

        try:
            # --- THIS IS THE FIX ---
            # Use the CustomEncoder to handle all special types
            insights_json = json.dumps(insights, indent=2, cls=CustomEncoder)
            # --- END FIX ---
        except Exception as e:
            print(f"Error serializing insights: {e}")
            # Fallback for serialization error
            insights_json = json.dumps({"error": "Failed to serialize insights"})

        prompt += "\n\nINSIGHTS:\n" + insights_json

        if evidence:
            prompt += "\n\nEVIDENCE SNIPPETS:\n" + "\n".join(evidence[:5])

        try:
            # Suppress any warnings from the API
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                resp = model.generate_content(prompt)
            return resp.text.strip()
        except Exception as e:
            return '[LLM generation failed] ' + str(e)

    # Fallback logic if Gemini is not available
    print("Falling back to non-LLM summary.")
    lines = []
    for p in insights.get('products', []):
        lines.append(
            f"{p['product_name']}: forecast mean {p.get('forecast_mean', 0):.1f}, risk_index {p.get('risk_index', 0.0):.2f}, P(stockout) {p.get('p_stockout', 0.0):.2f}.")
    recs = ["Check high-risk products and consider reordering.",
            "Investigate negative reviews for products with low sentiment."]
    return '\n'.join(lines + recs)
