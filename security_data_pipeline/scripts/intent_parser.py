# security_data_pipeline/scripts/intent_parser.py
import json, os
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise EnvironmentError(
        "❌ Missing OPENAI_API_KEY. Please set it using:\n"
        "   setx OPENAI_API_KEY your_key_here  (Windows)\n"
        "   export OPENAI_API_KEY=your_key_here  (Mac/Linux)"
    )

client = OpenAI(api_key=api_key)

def parse_intent(user_query: str):
    """Return {'intent': str, 'confidence': float}"""
    prompt = f"""
    Classify this query into one of:
    [crime_stats, alerts, urban_safety, demographics, news, map, other]
    and give confidence 0–1 with JSON only.
    Example: {{"intent":"crime_stats","confidence":0.82}}
    Query: {user_query}
    """

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        text = resp.choices[0].message.content.strip()
        data = json.loads(text)
        return data
    except Exception as e:
        print(f"[WARN] Intent parse failed: {e}")
        return {"intent": "other", "confidence": 0.0}