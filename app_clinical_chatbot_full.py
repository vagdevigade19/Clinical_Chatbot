import os
import re
import json
import csv
import io
import tempfile
import regex as regex_mod
from typing import List, Tuple, Union

import gradio as gr
from dotenv import load_dotenv

# LangChain core (v1.x)
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

# ===================== CONFIG =====================

load_dotenv()  # loads from .env if present

MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai").lower().strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
APP_TITLE = os.getenv(
    "APP_TITLE",
    "🩺 Clinical Chatbot (Symptom Intake, No Diagnosis)"
)

SYSTEM_POLICY = (
    "You are a clinical information assistant for preliminary health screening. "
    "Your responsibilities:\n"
    "1) Ask relevant questions about symptoms: onset, duration, severity (0–10), "
    "location, modifiers, associated symptoms, relevant history.\n"
    "2) Provide general, layperson-friendly information about possible categories of concern, "
    "STRICTLY without diagnosing any specific disease.\n"
    "3) Never prescribe medications, dosages, tests, or treatments.\n"
    "4) Encourage users to seek professional medical evaluation for any concerns.\n"
    "5) If red-flag symptoms suggest an emergency (e.g., chest pain, severe shortness of breath, "
    "one-sided weakness, confusion, uncontrolled bleeding, suicidal intent), clearly advise "
    "immediate emergency care.\n"
    "6) Be concise, supportive, and avoid fear-mongering. No hallucinated facts."
)

SUMMARY_PROMPT = (
    "Based on the entire conversation so far, write a concise, patient-friendly summary with:\n"
    "1) Reported symptoms (onset, duration, severity, location, associated symptoms, relevant history)\n"
    "2) Possible categories of concern (very high-level, non-diagnostic)\n"
    "3) Red flags to watch for, if any\n"
    "4) A short bullet list: 'What to tell a clinician'\n"
    "5) Next steps: emphasize consulting a clinician; mention emergency care if symptoms worsen or red flags occur.\n"
    "Do NOT provide any diagnosis. Do NOT recommend specific drugs or dosages."
)

DISCLAIMER = (
    "**Note:** This assistant provides general information only and cannot diagnose or treat. "
    "Always consult a licensed clinician for medical advice."
)

# ===================== SAFETY / PII =====================

PII_PATTERNS = [
    (r"\b(MR|MRN|ID|Patient ID)[:\s]*[A-Za-z0-9\-]+\b", "[REDACTED-ID]"),
    (r"\bDOB[:\s]*\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b", "[REDACTED-DOB]"),
    (r"\b[A-Z][a-z]+,\s[A-Z][a-z]+\b", "[REDACTED-NAME]"),               # Last, First
    (r"\b[A-Z][a-z]+(\s[A-Z][a-z]+){1,2}\b", "[REDACTED-NAME]"),        # John / John Doe / John P Doe
]

RED_FLAG_REGEX = regex_mod.compile(
    r"(chest pain|pressure in chest|severe shortness of breath|trouble breathing|"
    r"one[-\s]?sided weakness|face droop|slurred speech|confusion|fainting|"
    r"uncontrolled bleeding|seizure|blue lips|severe allergic reaction|anaphylaxis|"
    r"suicidal|homicidal|intent to harm|overdose|poison)",
    flags=regex_mod.I,
)

def redact_pii(text: str) -> str:
    text = text or ""
    for pat, repl in PII_PATTERNS:
        text = re.sub(pat, repl, text)
    return text

def is_emergency(text: str) -> bool:
    return bool(RED_FLAG_REGEX.search(text or ""))

def strip_html(s: str) -> str:
    return re.sub(r"<.*?>", "", s or "")

# ===================== MODEL FACTORY =====================

def make_llm():
    if MODEL_PROVIDER == "ollama":
        return ChatOllama(model=OLLAMA_MODEL, temperature=0.4)
    # default: OpenAI
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is missing. Set it in your .env or environment.")
    return ChatOpenAI(model=OPENAI_MODEL, temperature=0.4, api_key=OPENAI_API_KEY)

# ===================== SLOT-FILLING =====================

def missing_basics(user_text: str, chat_history: List[Tuple[str, str]]) -> List[str]:
    """Check if core clinical intake fields are missing."""
    user_l = (user_text or "").lower()
    joined = " ".join(
        [strip_html(u).lower() + " " + strip_html(a).lower() for (u, a) in chat_history]
    )

    def has_any(keys):
        return any(k in joined or k in user_l for k in keys)

    missing = []
    if not has_any(["since", "for ", "day", "week", "month", "hour", "duration"]):
        missing.append("duration (e.g., 'for 3 days')")
    if not has_any(["started", "onset", "began", "sudden", "gradual"]):
        missing.append("onset (sudden or gradual)")
    if not has_any(["mild", "moderate", "severe", "severity", "scale", "10/"]):
        missing.append("severity (0–10)")
    if not has_any(["left", "right", "upper", "lower", "chest", "throat", "head",
                    "abdomen", "back", "leg", "arm", "location"]):
        missing.append("location of the symptom")
    if not has_any(["also", "with", "associated", "together with", "plus",
                    "other symptoms"]):
        missing.append("any associated symptoms")
    return missing

# ===================== EXPORT HELPERS =====================

def export_json(chat_history: List[Tuple[str, str]]):
    """Create JSON transcript file; return update for gr.File."""
    if not chat_history:
        return gr.update(value=None, visible=False)

    data = []
    for u, a in chat_history:
        data.append({
            "user": strip_html(u),
            "assistant": strip_html(a),
        })

    fd, path = tempfile.mkstemp(prefix="transcript_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return gr.update(value=path, visible=True)

def export_csv(chat_history: List[Tuple[str, str]]):
    """Create CSV transcript file; return update for gr.File."""
    if not chat_history:
        return gr.update(value=None, visible=False)

    fd, path = tempfile.mkstemp(prefix="transcript_", suffix=".csv")
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user", "assistant"])
        for u, a in chat_history:
            writer.writerow([strip_html(u), strip_html(a)])

    return gr.update(value=path, visible=True)

# ===================== CHAT LOGIC =====================

def build_message_history(message_history: List[Union[SystemMessage, HumanMessage, AIMessage]]):
    """Ensure system message is first."""
    if not message_history or not isinstance(message_history[0], SystemMessage):
        return [SystemMessage(content=SYSTEM_POLICY), *message_history]
    return message_history

def chat_step(
    user_text: str,
    chat_history: List[Tuple[str, str]],
    message_history: List[Union[SystemMessage, HumanMessage, AIMessage]],
):
    user_text = redact_pii((user_text or "").strip())
    if not user_text:
        # Ask user to enter something
        info = "Please describe your symptoms or health concern."
        chat_history.append(("", info))
        return chat_history, message_history

    # 1) Emergency detection
    if is_emergency(user_text):
        msg = (
            "⚠️ Your description may indicate a serious or urgent symptom. "
            "If you are experiencing severe or worsening symptoms, please call your local "
            "emergency number (e.g., 911) or seek immediate emergency care."
        )
        chat_history.append((user_text, msg))
        message_history.extend([
            HumanMessage(content=user_text),
            AIMessage(content=msg),
        ])
        return chat_history, message_history

    # 2) Slot-filling hint
    need = missing_basics(user_text, chat_history)
    if need:
        prompt = (
            "Thanks for sharing. To better understand your situation, could you also tell me: "
            + ", ".join(need[:-1])
            + (" and " if len(need) > 1 else "")
            + need[-1]
            + "?"
        )
        reply = f"{prompt}\n\n{DISCLAIMER}"
        chat_history.append((user_text, reply))
        message_history.extend([
            HumanMessage(content=user_text),
            AIMessage(content=reply),
        ])
        return chat_history, message_history

    # 3) Normal LLM response
    llm = make_llm()
    enriched_history = build_message_history(message_history.copy())
    enriched_history.append(HumanMessage(content=user_text))

    resp = llm.invoke(enriched_history)
    ai_text = (resp.content or "").strip() + f"\n\n{DISCLAIMER}"

    chat_history.append((user_text, ai_text))
    message_history.extend([
        HumanMessage(content=user_text),
        AIMessage(content=ai_text),
    ])
    return chat_history, message_history

def summarize_session(
    chat_history: List[Tuple[str, str]],
    message_history: List[Union[SystemMessage, HumanMessage, AIMessage]],
):
    if not chat_history:
        return "No conversation yet to summarize."

    llm = make_llm()
    enriched = build_message_history(message_history.copy())
    enriched.append(SystemMessage(content=SUMMARY_PROMPT))

    resp = llm.invoke(enriched)
    summary = (resp.content or "").strip()
    summary += (
        "\n\n**Disclaimer:** This summary is informational and not a medical diagnosis. "
        "Please consult a licensed clinician."
    )
    return summary

def clear_all():
    return [], []  # chat_history, message_history reset

# ===================== UI (Gradio) =====================

with gr.Blocks(title=APP_TITLE) as demo:
    gr.Markdown(
        f"# {APP_TITLE}\n"
        "This assistant helps collect and summarize symptoms in plain language.\n\n"
        "**It does NOT provide medical advice, diagnosis, or prescriptions.**"
    )

    with gr.Row():
        gr.Label(label="Model Backend", value=MODEL_PROVIDER.upper())
        gr.Label(
            label="Model",
            value=(OPENAI_MODEL if MODEL_PROVIDER == "openai" else OLLAMA_MODEL),
        )

    chatbot = gr.Chatbot(
        label="Conversation",
        type="tuples",   # list of (user, assistant)
        height=420,
    )

    user_box = gr.Textbox(
        placeholder="Describe your symptoms (e.g., 'Sore throat for 3 days with mild fever').",
        lines=3,
        label="Your message",
    )

    with gr.Row():
        send_btn = gr.Button("Send", variant="primary")
        summarize_btn = gr.Button("Summarize Session")
        clear_btn = gr.Button("Clear Conversation")

    with gr.Row():
        dl_json_btn = gr.Button("Download JSON Transcript")
        dl_csv_btn = gr.Button("Download CSV Transcript")

    summary_out = gr.Textbox(
        label="Session Summary (Patient-friendly)",
        lines=14,
    )

    json_file = gr.File(label="JSON Transcript", visible=False)
    csv_file = gr.File(label="CSV Transcript", visible=False)

    gr.Markdown(
        "<div style='opacity:0.9; font-size: 0.9em;'>"
        "<strong>Important:</strong> This chatbot is for educational and informational purposes only. "
        "It does not replace professional medical evaluation. "
        "If you may be experiencing an emergency, call your local emergency number immediately."
        "</div>"
    )

    # State: we keep structured history separately for LLM context
    chat_state = gr.State([])        # List[Tuple[user, assistant]]
    msg_state = gr.State([])         # List[System/Human/AI messages]

    # Wiring
    send_btn.click(
        chat_step,
        inputs=[user_box, chat_state, msg_state],
        outputs=[chatbot, msg_state],
    ).then(
        lambda _msg: "",
        inputs=user_box,
        outputs=user_box,
    )

    user_box.submit(
        chat_step,
        inputs=[user_box, chat_state, msg_state],
        outputs=[chatbot, msg_state],
    ).then(
        lambda _msg: "",
        inputs=user_box,
        outputs=user_box,
    )

    summarize_btn.click(
        summarize_session,
        inputs=[chat_state, msg_state],
        outputs=[summary_out],
    )

    clear_btn.click(
        clear_all,
        outputs=[chat_state, msg_state],
    ).then(
        lambda: gr.update(value=[]),
        outputs=[chatbot],
    ).then(
        lambda: "",
        outputs=[summary_out],
    ).then(
        lambda: gr.update(value=None, visible=False),
        outputs=[json_file],
    ).then(
        lambda: gr.update(value=None, visible=False),
        outputs=[csv_file],
    )

    dl_json_btn.click(
        export_json,
        inputs=[chat_state],
        outputs=[json_file],
    )

    dl_csv_btn.click(
        export_csv,
        inputs=[chat_state],
        outputs=[csv_file],
    )

if __name__ == "__main__":
    demo.launch()

