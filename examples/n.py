#!/usr/bin/env python3
"""
eDOSTHI Longitudinal Chat Runner
"""

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
import argparse
import requests


from annotate.graph.runner import run_patient_sessions
from annotate.agents.memory.patient_memory import PatientMemory
from annotate.utils.logger import (
    configure_ollama_text_log,
    configure_run_logging,
    configure_run_trace,
    log_data,
    log_ollama_exchange,
)
# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data/sessions/sessions.json"
RUN_LOG_DIR = PROJECT_ROOT / "data/logs/runs"

OLLAMA_URL = "http://localhost:11434/api/chat"
#OLLAMA_MODEL = "wmb/llamasupport"
OLLAMA_MODEL = "gemma3:4b"
EXIT_CMDS = {"exit", "quit"}

# ---------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------

SYSTEM_PROMPT_INITIAL = """
You are eDOSTHI, a motivational interviewing chatbot helping users quit smoking.
Use open-ended questions, reflection, affirmation.
Keep responses short (<30 words).
"""

SYSTEM_PROMPT_FOLLOWUP = """
You are eDOSTHI, continuing a follow-up counseling session.

Previous clinical summary:
{reasoning_summary}

Use motivational interviewing.
Keep responses short (<30 words).
"""

SYSTEM_PROMPT_INITIAL_BN = """
আপনি eDOSTHI, ধূমপান ছাড়তে সাহায্য করার জন্য একটি motivational interviewing chatbot।
খোলা প্রশ্ন, প্রতিফলন, এবং উৎসাহ ব্যবহার করুন।
উত্তর ছোট রাখুন (<৩০ শব্দ)।
বাংলায় উত্তর দিন।
"""

SYSTEM_PROMPT_FOLLOWUP_BN = """
আপনি eDOSTHI, একটি follow-up counselling session চালিয়ে যাচ্ছেন।

আগের clinical summary:
{reasoning_summary}

motivational interviewing ব্যবহার করুন।
উত্তর ছোট রাখুন (<৩০ শব্দ)।
বাংলায় উত্তর দিন।
"""

SYSTEM_PROMPTS = {
    "en": {
        "initial": SYSTEM_PROMPT_INITIAL,
        "followup": SYSTEM_PROMPT_FOLLOWUP,
        "context_label": "PATIENT CONTEXT",
        "stage_trajectory": "Stage trajectory",
        "persistent_triggers": "Persistent triggers",
        "persistent_concerns": "Persistent concerns",
        "relapse_count": "Relapse count",
    },
    "bn": {
        "initial": SYSTEM_PROMPT_INITIAL_BN,
        "followup": SYSTEM_PROMPT_FOLLOWUP_BN,
        "context_label": "রোগীর প্রসঙ্গ",
        "stage_trajectory": "স্টেজ ট্র্যাজেক্টরি",
        "persistent_triggers": "বারবার দেখা দেওয়া ট্রিগার",
        "persistent_concerns": "বারবার দেখা দেওয়া উদ্বেগ",
        "relapse_count": "রিল্যাপ্স সংখ্যা",
    },
}

# ---------------------------------------------------------------------
# DB Helpers
# ---------------------------------------------------------------------

def load_db():
    if DB_PATH.exists():
        return json.loads(DB_PATH.read_text(encoding="utf-8"))
    return {}


def save_db(db):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    DB_PATH.write_text(json.dumps(db, indent=2, ensure_ascii=False), encoding="utf-8")


def load_session(session_id):
    return load_db().get(session_id)


def save_session(session_id, record):
    db = load_db()
    db[session_id] = record
    save_db(db)

# ---------------------------------------------------------------------
# Prompt Builder
# ---------------------------------------------------------------------

def build_system_prompt(session, profile, language: str = "en"):
    prompt_config = SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS["en"])

    if profile.total_sessions == 0:
        return prompt_config["initial"]

    last_summary = session.get("last_result", {}).get("reasoning_summary", "unknown")

    return prompt_config["followup"].format(
        reasoning_summary=last_summary
    ) + f"""

{prompt_config["context_label"]}:
- {prompt_config["stage_trajectory"]}: {' → '.join(profile.stage_trajectory)}
- {prompt_config["persistent_triggers"]}: {profile.persistent_triggers}
- {prompt_config["persistent_concerns"]}: {profile.persistent_concerns}
- {prompt_config["relapse_count"]}: {profile.relapse_count}
"""

# ---------------------------------------------------------------------
# Chat Loop
# ---------------------------------------------------------------------

def chat_loop(system_prompt: str) -> List[dict]:
    messages: List[dict] = []

    print("\neDOSTHI: Hello! I'm here to help you quit smoking.")
    print('(Type "exit" to end session)\n')

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in EXIT_CMDS:
            print("\neDOSTHI: Take care. You're making progress.")
            break

        messages.append({"role": "user", "content": user_input})

        try:
            payload = {
                "model": OLLAMA_MODEL,
                "messages": [{"role": "system", "content": system_prompt}] + messages,
                "stream": False,
            }

            response = requests.post(OLLAMA_URL, json=payload, timeout=60)
            response.raise_for_status()
            reply = response.json()["message"]["content"].strip()
            log_ollama_exchange(
                model=OLLAMA_MODEL,
                temperature=0.0,
                prompt=json.dumps(payload, indent=2, ensure_ascii=False),
                response_text=reply,
                attempt=1,
                status="chat_success",
            )

        except Exception as e:
            reply = f"[Model error: {e}]"
            log_ollama_exchange(
                model=OLLAMA_MODEL,
                temperature=0.0,
                prompt=json.dumps(payload, indent=2, ensure_ascii=False) if 'payload' in locals() else '',
                response_text=reply,
                attempt=1,
                status="chat_exception",
            )

        print(f"\neDOSTHI: {reply}\n")
        messages.append({"role": "assistant", "content": reply})

    return messages

# ---------------------------------------------------------------------
# Main Chat Runner
# ---------------------------------------------------------------------

def run_chat(session_id: Optional[str] = None, enable_debate: bool = False, language: str = "en"):

    # Session init
    if session_id is None:
        session_id = str(uuid.uuid4())
        session = None
        print(f"New patient session: {session_id}")
    else:
        session = load_session(session_id)
        print(f"Resuming patient: {session_id}")

    run_stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_log_path = configure_run_logging(RUN_LOG_DIR / f"{session_id}_{run_stamp}.log")
    run_trace_path = configure_run_trace(
        RUN_LOG_DIR / f"{session_id}_{run_stamp}.json",
        metadata={"session_id": session_id, "started_at": run_stamp, "language": language},
    )
    ollama_text_path = configure_ollama_text_log(RUN_LOG_DIR / f"{session_id}_{run_stamp}_ollama.txt")
    print(f"Detailed pipeline log: {run_log_path}")
    print(f"Structured trace JSON: {run_trace_path}")
    print(f"Ollama full text log: {ollama_text_path}")
    print(f"Ollama latest full output: {RUN_LOG_DIR / 'ollama_full_output.txt'}")

    # Patient Memory
    pm = PatientMemory()
    profile = pm.load_or_create(session_id)
    print(profile)
    print("\n[Patient Context]")
    print(f"Total sessions: {profile.total_sessions}")
    if profile.stage_trajectory:
        print("Trajectory:", " → ".join(profile.stage_trajectory))

    # Prompt
    system_prompt = build_system_prompt(session or {}, profile, language=language)

    print("\n" + "=" * 60)
    print("SYSTEM PROMPT")
    print("=" * 60)
    print(system_prompt)
    print("=" * 60)

    # Chat
    messages = chat_loop(system_prompt)

    if not messages:
        print("No conversation. Exiting.")
        return

    # Transcript (FIXED ROLES)
    transcript = "\n".join(
        f"{'Patient' if m['role']=='user' else 'Counselor'}: {m['content']}"
        for m in messages
    )
    log_data("run_chat.transcript", {"session_id": session_id, "transcript": transcript})

    # Compute turn BEFORE pipeline (FIX)
    turn = (session or {}).get("turn", 0) + 1

    # Pipeline
    print("\nRunning longitudinal pipeline...\n")

    results = run_patient_sessions(
        sessions=[{
            "id": f"{session_id}_turn_{turn}",
            "conversation": transcript
        }],
        patient_id=session_id,
        patient_memory=pm,
        enable_debate=enable_debate
    )

    result = results[0]
    log_data("run_chat.final_result", result)

    # Save session
    history = (session or {}).get("history", [])
    history.append({
        "turn": turn,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "messages": messages,
        "result": result
    })

    save_session(session_id, {
        "session_id": session_id,
        "turn": turn,
        "last_result": result,
        "history": history,
        "updated_at": datetime.now(timezone.utc).isoformat()
    })

    print(f"\nSession saved: {session_id}")

    # Output
    row = result.get("dataset_row", {})
    long_o = result.get("longitudinal_output", {})

    # FIXED delta handling
    delta = long_o.get("delta")

    print("\n" + "=" * 60)
    print("SESSION RESULT")
    print("=" * 60)

    print("Stage:", row.get("stage_of_change"))
    print("Concerns:", ", ".join(row.get("all_concerns", [])) or "none")
    print("Triggers:", ", ".join(row.get("all_triggers", [])) or "none")
    print("Confidence:", row.get("confidence_score"))

    print("\n--- Longitudinal ---")

    if long_o.get("stage_trajectory"):
        print("Trajectory:", " → ".join(long_o["stage_trajectory"]))

    if delta:
        if delta.get("stage_direction"):
            print("Stage change:", delta["stage_direction"])

        if delta.get("clinical_summary"):
            print("Clinical summary:", delta["clinical_summary"])
    else:
        print("First session → no longitudinal delta")

    flags = long_o.get("longitudinal_flags", [])
    if flags:
        print("\n⚠ Flags:")
        for f in flags:
            print("-", f)
    else:
        print("\n✓ Stable progression")

    print("\nContinue with:")
    print(f"python run_chat.py --session-id {session_id}")

# ---------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--enable-debate", action="store_true", help="Enable debate loop for each session") 
    language_group = parser.add_mutually_exclusive_group()
    language_group.add_argument("-en", dest="language", action="store_const", const="en", help="Use English chat prompts")
    language_group.add_argument("-bn", dest="language", action="store_const", const="bn", help="Use Bengali chat prompts")
    parser.set_defaults(language="en")
    args = parser.parse_args()
    run_chat(args.session_id, enable_debate=args.enable_debate, language=args.language)
