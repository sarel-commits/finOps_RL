"""Prompts — verbatim copies of the baseline eval prompts (scripts/agent.py SYSTEM_NATIVE, scripts/run_bench.py
ITB_PROMPT, v1). Training on the exact eval prompt keeps train and eval aligned; tests/test_prompts.py fails if
these drift from the eval harness.
"""
import json

SYSTEM = (
    "You are an expert FinOps (cloud financial operations) analyst agent. You can execute Python with the run_python tool "
    "to inspect the data files in your working directory. Always compute answers from the data with code; never guess or "
    "invent numbers or identifiers. When you are done, give your final answer as a single JSON object wrapped in "
    "<answer></answer> tags, exactly matching the required format, and do not call any tool in that final message."
)

TASK = """A cost anomaly was detected in a cloud billing dataset. Your job is root cause analysis: identify the resource(s) responsible for the anomalous cost change.

Anomaly details (from anomaly.json, also in your working directory):
{anomaly}

The file data.csv in your working directory contains the daily cost data for the accounts. Investigate it with code (inspect the columns first) and determine which resource(s) caused the anomalous cost for this account on this date. A resource is identified by its name (the exact value as it appears in data.csv) and its type (the data.csv column that value comes from, written in title case with spaces instead of underscores, e.g. a value from a column named some_column has type "Some Column").

Required final answer format (JSON inside <answer></answer> tags):
{{"resource": [{{"name": "<resource name>", "type": "<resource type>"}}]}}
List every resource that is a root cause; include only root causes."""

RUN_PYTHON_DESCRIPTION = (
    "Execute a Python 3.12 script in the task's working directory (which contains the data files). "
    "pandas and numpy are installed. Returns combined stdout/stderr (truncated to 8000 chars). "
    "Each call is a fresh process: re-load data every time. Use print() to see results.")

RUN_PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": RUN_PYTHON_DESCRIPTION,
        "parameters": {"type": "object",
                       "properties": {"code": {"type": "string", "description": "Python source code to run"}},
                       "required": ["code"]},
    },
}


def anomaly_text(anomaly: dict) -> str:
    """Same text the eval reads from anomaly.json (ITBench files use indent=4)."""
    return json.dumps(anomaly, indent=4)


def task_prompt(anomaly: dict) -> str:
    return TASK.format(anomaly=anomaly_text(anomaly))
