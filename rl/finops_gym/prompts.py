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


# ---------------------------------------------------------------- improved harness (eval configs *_nudges_v2)
# Verbatim copies of scripts/agent.py SYSTEM_NATIVE_V2 / NUDGE_* and scripts/run_bench.py ITB_METHOD_V2;
# tests/test_env.py::test_prompts_match_eval_harness checks they have not drifted.
SYSTEM_V2 = """You are an expert FinOps (cloud financial operations) analyst agent. You can execute Python with the run_python tool to inspect the data files in your working directory. Always compute answers from the data with code; never guess or invent numbers or identifiers.

You have a strict budget of {max_steps} tool calls. Work efficiently:
- Plan your analysis before writing code, and do several computations in one tool call when you can.
- Do not re-run an analysis that you have already done with small variations. If a result already answers the question, stop and answer.
- An empty result is a valid answer. If the data shows that nothing qualifies, say so; do not keep searching.
- It is always better to give your best-supported answer than to run out of steps without one.

When you are done, give your final answer as a single JSON object wrapped in <answer></answer> tags, exactly matching the required format, and do not call any tool in that final message."""

METHOD_V2 = """Recommended method (aim to finish in about 6-8 tool calls):
1. Inspect the columns and filter to the anomalous account. Compare the anomaly date with a baseline of the preceding normal days.
2. For EVERY categorical column (not only one), compute the cost change per value between the baseline and the anomaly date. Do this for all columns in a single tool call.
3. Decide which column is the root-cause dimension: the one whose values explain the total cost change most fully and most specifically. A change in one column often shows up indirectly in other columns, so compare the columns before choosing; do not assume a particular column.
4. The root cause can be one value or many values of that column. Include every value that materially drives the change, and nothing else.
5. Answer.

"""
_SPLIT = "Required final answer format"
TASK_V2 = TASK.replace(_SPLIT, METHOD_V2 + _SPLIT, 1)
assert TASK_V2.count(_SPLIT) == 1 and TASK_V2 != TASK

NUDGE_STEP = "[Step {k} of {max_steps} used.]"
NUDGE_WARN = ("[Budget warning: only 5 tool calls remain. Finish the analysis you need and give your final answer soon. "
              "Do not start new exploratory analyses.]")
NUDGE_FINAL = ("[The tool budget is exhausted. You cannot run more code. Based on the analysis you have already done, "
               "give your best final answer now as JSON inside <answer></answer> tags.]")


def system_prompt(harness: str = "v1", max_steps: int = 20) -> str:
    return SYSTEM_V2.format(max_steps=max_steps) if harness == "improved" else SYSTEM


def task_prompt_for(anomaly: dict, harness: str = "v1") -> str:
    return (TASK_V2 if harness == "improved" else TASK).format(anomaly=anomaly_text(anomaly))


def nudge(steps_used: int, max_steps: int) -> str:
    """Suffix the eval harness appends to the LAST tool output of each tool turn (budget_nudges=True)."""
    s = "\n\n" + NUDGE_STEP.format(k=steps_used, max_steps=max_steps)
    if max_steps - steps_used == 5:
        s += "\n" + NUDGE_WARN
    return s
