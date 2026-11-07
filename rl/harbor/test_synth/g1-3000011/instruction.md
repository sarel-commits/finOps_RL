A cost anomaly was detected in a cloud billing dataset. Your job is root cause analysis: identify the resource(s) responsible for the anomalous cost change.

Anomaly details (from anomaly.json, also in your working directory):
{
    "date": "2025-08-10",
    "account_id": "1GM9Oyk"
}

The file data.csv in your working directory contains the daily cost data for the accounts. Investigate it with code (inspect the columns first) and determine which resource(s) caused the anomalous cost for this account on this date. A resource is identified by its name (the exact value as it appears in data.csv) and its type (the data.csv column that value comes from, written in title case with spaces instead of underscores, e.g. a value from a column named some_column has type "Some Column").

Required final answer format (JSON inside <answer></answer> tags):
{"resource": [{"name": "<resource name>", "type": "<resource type>"}]}
List every resource that is a root cause; include only root causes.

You are working in /app (anomaly.json and data.csv are there). Python 3.12 with pandas and numpy is installed. When you are done, write your final answer as a single JSON object in exactly the format above to /app/answer.json (no <answer> tags in the file).
