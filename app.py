"""
AI Utility Hub - Flask backend

Run:
    pip install -r requirements.txt
    set ANTHROPIC_API_KEY=your-key-here        (Windows CMD)
    $env:ANTHROPIC_API_KEY="your-key-here"     (Windows PowerShell)
    export ANTHROPIC_API_KEY=your-key-here     (Mac / Linux)
    python app.py

If no API key is set (or the internet is down) every tool automatically
falls back to the built-in rule-based logic, so the demo never breaks.
"""

import os
import re

from flask import Flask, jsonify, render_template, request

try:
    import anthropic
except ImportError:  # the app still runs without the SDK (offline mode)
    anthropic = None


app = Flask(__name__)


# -----------------------------
# AI SETUP
# -----------------------------

API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Model name can be changed without editing code: set AI_MODEL in the environment.
MODEL = os.environ.get("AI_MODEL", "claude-sonnet-5-5")

client = None
if anthropic is not None and API_KEY:
    client = anthropic.Anthropic(api_key=API_KEY, timeout=20.0, max_retries=1)

MAX_CHAT_LENGTH = 1000
MAX_TEXT_LENGTH = 2000
MAX_CODE_LENGTH = 4000

SUPPORTED_LANGUAGES = ["Hindi", "Spanish", "French", "Marathi", "German", "Japanese"]

CHAT_SYSTEM = (
    "You are the friendly assistant inside AI Utility Hub, a student-built web app "
    "with speech-to-text, text-to-speech, translation, code explanation and chat tools. "
    "Answer clearly and briefly (under 120 words). Reply in plain text only: "
    "no Markdown, no asterisks, no headings."
)

CODE_SYSTEM = (
    "You are a patient programming tutor for beginners. The user will paste code in "
    "any language. Reply in plain text only (no Markdown, no code fences). Use this "
    "layout: a line 'Simple explanation:', then one bullet per meaningful line or "
    "block that starts with '•' (quote the code, then '→', then a plain-English "
    "explanation), then a blank line, then a line 'Pseudocode:' followed by short "
    "steps. Keep it under 300 words. If the input is not code, say so in one sentence."
)


def ai_available():
    return client is not None


def ask_ai(system, messages, max_tokens=700):
    """Send a request to the AI model and return plain text.
    Raises an exception on any problem so the caller can use its fallback."""
    if client is None:
        raise RuntimeError("AI is not configured")

    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
    )

    text = "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()

    if not text:
        raise RuntimeError("Empty AI response")

    return text


def get_json_body():
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/status")
def status():
    return jsonify({"ai": ai_available()})


# -----------------------------
# CHATBOT
# -----------------------------

def rule_based_reply(message):
    """Offline fallback. Matches whole words so 'this' or 'something'
    is no longer mistaken for 'hi'."""
    text = message.lower().strip()
    words = set(re.findall(r"[a-z']+", text))

    if "how are you" in text:
        return "I'm doing great! 🤖 Thanks for asking."

    if "who are you" in text or "your name" in text:
        return "I'm the AI Utility Hub chatbot."

    if words & {"hello", "hi", "hey"}:
        return "Hello! 👋 How can I help you today?"

    if "python" in words:
        return (
            "Python is a beginner-friendly programming language commonly used "
            "for web development, AI, automation and data science."
        )

    if "flask" in words:
        return (
            "Flask is a lightweight Python web framework used for building "
            "web applications and APIs."
        )

    if words & {"thank", "thanks"}:
        return "You're welcome! 😊"

    if words & {"bye", "goodbye"}:
        return "Goodbye! Have a great day! 👋"

    return (
        "That's interesting! I'm still learning. "
        "Try asking me about Python, Flask, programming or AI."
    )


def clean_history(raw):
    """Keep only the last few valid chat turns sent by the browser."""
    history = []

    if not isinstance(raw, list):
        return history

    for item in raw[-10:]:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            history.append({"role": role, "content": content.strip()[:MAX_CHAT_LENGTH]})

    # The conversation sent to the AI must start with a user message.
    while history and history[0]["role"] != "user":
        history.pop(0)

    return history


@app.route("/api/chat", methods=["POST"])
def chat():
    data = get_json_body()

    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"reply": "Please type something first.", "source": "system"})

    message = message[:MAX_CHAT_LENGTH]

    if ai_available():
        try:
            history = clean_history(data.get("history"))
            reply = ask_ai(
                CHAT_SYSTEM,
                history + [{"role": "user", "content": message}],
                max_tokens=400,
            )
            return jsonify({"reply": reply, "source": "ai"})
        except Exception as error:
            app.logger.warning("AI chat failed, using fallback: %s", error)

    return jsonify({"reply": rule_based_reply(message), "source": "fallback"})


# -----------------------------
# CODE EXPLANATION
# -----------------------------

def is_assignment(line):
    """True for 'x = 5' or 'x += 1', but not for 'x == 5'."""
    return re.match(r"^[\w\.\[\]]+\s*([+\-*/%]|//|\*\*)?=(?!=)", line) is not None


def describe_line(line):
    """Return (plain explanation, pseudocode) for one line of code."""
    if line.startswith("#"):
        return "This is a comment. It is ignored when the program runs.", None

    if line.startswith("print("):
        return "This displays something on the screen.", "DISPLAY the required output"

    if "input(" in line:
        return "This takes input from the user.", "TAKE input from the user"

    if line.startswith("elif "):
        return (
            "This checks another condition if the earlier ones were false.",
            "OTHERWISE IF the condition is true",
        )

    if line.startswith("if "):
        return "This checks a condition.", "IF the condition is true"

    if line.startswith("else"):
        return (
            "This runs when none of the earlier conditions were true.",
            "OTHERWISE",
        )

    if line.startswith("for "):
        return "This creates a loop that repeats code.", "REPEAT the following steps"

    if line.startswith("while "):
        return (
            "This creates a loop that continues while a condition is true.",
            "WHILE the condition is true, repeat",
        )

    if line.startswith("def "):
        return "This defines a function.", "DEFINE a function"

    if line.startswith("class "):
        return "This defines a class (a blueprint for objects).", "DEFINE a class"

    if line.startswith("return"):
        return "This sends a value back from the function.", "RETURN the result"

    if line.startswith("import ") or line.startswith("from "):
        return "This loads extra code (a library) to use in the program.", "LOAD the required library"

    if line.startswith("try"):
        return "This starts a block that handles possible errors.", "TRY the following steps"

    if line.startswith("except"):
        return "This runs if an error happened in the try block.", "IF an error occurs"

    if is_assignment(line):
        return "This assigns a value to a variable.", "STORE a value in a variable"

    return "This is a line of program code.", "PERFORM the required operation"


def explain_simple_code(code):
    """Offline fallback: explains code line by line using simple rules."""
    lines = [line.strip() for line in code.splitlines() if line.strip()]

    explanation = ["Simple explanation:\n"]
    pseudocode = []

    for line in lines:
        meaning, pseudo = describe_line(line)
        explanation.append(f"• {line} → {meaning}")

        if pseudo:
            pseudocode.append(pseudo)

    explanation.append("\nPseudocode:")
    explanation.extend(pseudocode)

    return "\n".join(explanation)


@app.route("/api/explain-code", methods=["POST"])
def explain_code():
    data = get_json_body()

    code = str(data.get("code", "")).strip()

    if not code:
        return jsonify({"result": "Please enter some code first.", "source": "system"})

    code = code[:MAX_CODE_LENGTH]

    if ai_available():
        try:
            result = ask_ai(
                CODE_SYSTEM,
                [{"role": "user", "content": code}],
                max_tokens=900,
            )
            return jsonify({"result": result, "source": "ai"})
        except Exception as error:
            app.logger.warning("AI code explanation failed, using fallback: %s", error)

    return jsonify({"result": explain_simple_code(code), "source": "fallback"})


# -----------------------------
# TRANSLATION
# -----------------------------

# Offline fallback: a few common phrases only.
FALLBACK_TRANSLATIONS = {
    "Hindi": {
        "hello": "नमस्ते",
        "hello world": "नमस्ते दुनिया",
        "how are you": "आप कैसे हैं?",
        "good morning": "सुप्रभात",
        "thank you": "धन्यवाद",
    },
    "Spanish": {
        "hello": "hola",
        "hello world": "hola mundo",
        "how are you": "¿Cómo estás?",
        "good morning": "buenos días",
        "thank you": "gracias",
    },
    "French": {
        "hello": "bonjour",
        "hello world": "bonjour le monde",
        "how are you": "comment allez-vous ?",
        "good morning": "bonjour",
        "thank you": "merci",
    },
}


def fallback_translate(text, language):
    # Ignore capital letters and punctuation so "How are you?" still matches.
    cleaned = re.sub(r"[^\w\s]", "", text.lower()).strip()

    phrases = FALLBACK_TRANSLATIONS.get(language, {})

    if cleaned in phrases:
        return phrases[cleaned]

    return (
        f"Offline mode: only a few common phrases (like 'hello' or 'thank you') "
        f"can be translated to {language}. Connect an API key for full translation."
    )


@app.route("/api/translate", methods=["POST"])
def translate():
    data = get_json_body()

    text = str(data.get("text", "")).strip()
    language = str(data.get("language", "")).strip()

    if not text:
        return jsonify({"translation": "Please enter some text.", "source": "system"})

    if language not in SUPPORTED_LANGUAGES:
        return jsonify({"translation": "Please choose a supported language.", "source": "system"})

    text = text[:MAX_TEXT_LENGTH]

    if ai_available():
        try:
            result = ask_ai(
                f"Translate the user's text into {language}. "
                "Reply with only the translation: no quotes, notes or explanations.",
                [{"role": "user", "content": text}],
                max_tokens=700,
            )
            return jsonify({"translation": result, "source": "ai"})
        except Exception as error:
            app.logger.warning("AI translation failed, using fallback: %s", error)

    return jsonify({"translation": fallback_translate(text, language), "source": "fallback"})


# -----------------------------
# RUN APPLICATION
# -----------------------------

if __name__ == "__main__":
    # Debug mode is OFF by default (safer for a live demo).
    # To turn it on while developing:  set FLASK_DEBUG=1
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")