import os as _os

# Local-only POC: never send traces to hosted LangSmith unless the user opts in explicitly.
_os.environ.setdefault("LANGSMITH_TRACING", "false")
_os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
