"""Clinical conversation annotation library."""

__version__ = "0.1.0"

__all__ = [
    "AgentMemory",
    "PatientMemory",
    "run_pipeline",
    "run_session",
    "run_patient_sessions",
]


def __getattr__(name: str):
    if name == "AgentMemory":
        from annotate.agents.memory.agent_memory import AgentMemory

        return AgentMemory
    if name == "PatientMemory":
        from annotate.agents.memory.patient_memory import PatientMemory

        return PatientMemory
    if name in {"run_pipeline", "run_session", "run_patient_sessions"}:
        from annotate.graph.runner import run_patient_sessions, run_pipeline, run_session

        return {
            "run_pipeline": run_pipeline,
            "run_session": run_session,
            "run_patient_sessions": run_patient_sessions,
        }[name]
    raise AttributeError(f"module 'annotate' has no attribute {name!r}")
