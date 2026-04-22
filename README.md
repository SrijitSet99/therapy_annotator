# Annotate

`annotate` is an installable Python library for annotating tobacco-cessation counseling conversations into structured clinical training data. It uses a multi-agent pipeline with Ollama-backed extractors, confidence checks, revision, consensus, reasoning, sanity checks, and optional longitudinal patient memory.

## Features

- Single-session annotation with `run_pipeline`
- Longitudinal patient/session annotation with `run_session` and `run_patient_sessions`
- Pydantic output schemas for clinical and longitudinal annotations
- LangGraph workflow orchestration
- Prompt templates packaged with the library
- CLI helpers for single conversations, batch processing, and dataset generation
- Configurable local logging and patient-memory directories

## Installation

From this repository:

```bash
cd annotate
pip install -e .
```

For a normal local install:

```bash
cd annotate
pip install .
```

Install Ollama and pull the default model:

```bash
ollama serve
ollama pull gemma3:4b
```

## Requirements

Runtime Python dependencies are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

The package requires Python 3.10 or newer.

## Quick Start

```python
from annotate import run_pipeline

chat = """
Counselor: What makes quitting difficult right now?
Client: I smoke most when I feel stressed after work. I tried quitting last month but started again.
"""

row = run_pipeline(chat, conversation_id="demo-001")
print(row)
```

## Longitudinal Usage

```python
from annotate import PatientMemory, run_session

memory = PatientMemory(patients_dir="patients")

result = run_session(
    chat="Client: I cut down this week, but cravings after dinner are still hard.",
    patient_id="patient-001",
    conversation_id="session-001",
    patient_memory=memory,
)

print(result["dataset_row"])
print(result["longitudinal_output"])
```

## Command Line

After installation, these console commands are available:

```bash
annotate-single data/raw/sample_chat.txt
annotate-batch data/raw/conversations.json data/processed/output.json
annotate-generate data/raw/conversations.json data/processed/training_dataset.jsonl 4
```

You can also run the packaged examples as modules:

```bash
python -m annotate.examples.run_single_conversation data/raw/sample_chat.txt
python -m annotate.examples.run_batch_pipeline data/raw/conversations.json data/processed/output.json
python -m annotate.examples.run_multisession data/sessions/sessions.json
```

## Configuration

Default model settings live in `annotate/config/model_config.py`.

Important defaults:

- `MODEL_NAME`: `gemma3:4b`
- `OLLAMA_URL`: `http://localhost:11434/api/generate`
- `TEMPERATURE`: `0.2`
- `CONFIDENCE_DEBATE_THRESHOLD`: `0.8`
- `MAX_REVISION_ROUNDS`: `3`
- `CONSENSUS_VOTERS`: `3`

Runtime output locations can be overridden with environment variables:

```bash
export ANNOTATE_LOG_DIR=annotate_logs
export ANNOTATE_PATIENTS_DIR=data/patients
```

## Public API

```python
from annotate import AgentMemory, PatientMemory, run_patient_sessions, run_pipeline, run_session
```

`run_pipeline(chat, conversation_id=None, memory=None)` returns a dataset row dictionary for one conversation.

`run_session(chat, patient_id, conversation_id=None, patient_memory=None, enable_debate=False)` returns:

```python
{
    "dataset_row": {...},
    "longitudinal_output": {...},
    "session_number": 1,
    "patient_id": "patient-001",
}
```

`run_patient_sessions(sessions, patient_id, patient_memory=None, enable_debate=False)` runs multiple sessions sequentially for one patient.

## Expected Input

For batch processing, use a JSON list:

```json
[
  {
    "id": "conv-001",
    "conversation": "Counselor: ...\nClient: ..."
  }
]
```

## Notes

This package calls a local Ollama server during normal annotation. Importing the package does not call Ollama, but running the pipeline requires `ollama serve` and the configured model to be available.
