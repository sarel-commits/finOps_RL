---
license: apache-2.0
dataset_info:
  features:
    - name: scenario_id
      dtype: string
    - name: domain
      dtype: string
    - name: version
      dtype: string
    - name: task_type
      dtype: string
    - name: snapshot_data
      dtype: object
  configs:
    - config_name: sre
      data_files:
        - split: scenarios
          path: snapshots/sre/v0.2-B96DF826-4BB2-4B62-97AB-6D84254C53D7/Scenario-*/
    - config_name: finops
      data_files:
        - split: scenarios
          path: snapshots/finops/v0.1-finops-anomaly/Scenario-*/
    - config_name: ciso
      data_files:
        - split: scenarios
          path: snapshots/ciso/v0.1/k8s-opa-static-cis-*/
size_categories:
  - 10K<n<100K
task_categories:
  - question-answering
  - text-generation
  - other
language:
  - en
tags:
  - kubernetes
  - sre
  - finops
  - ciso
  - incident-response
  - fault-diagnosis
  - root-cause-analysis
  - observability
  - cost-optimization
  - llm-agents
  - it-automation
  - benchmarking
pretty_name: ITBench-Lite
---

# ITBench-Lite Dataset Card

## Dataset Overview

**Dataset Name**: ITBench-Lite  
**Organization**: IBM Research  
**License**: Apache 2.0  
**Language**: English  
**Paper**: [ITBench: Evaluating AI Agents across Diverse Real-World IT Automation Tasks](https://arxiv.org/pdf/2502.05352)  
**GitHub**: [ITBench](https://github.com/itbench-hub/ITBench)  

ITBench-Lite is a systematic framework for benchmarking LLMs and AI Agents on real-world IT automation tasks. This dataset contains **65 scenarios** across three critical domains:

- **Site Reliability Engineering (SRE)**: 35 scenarios with environment snapshots for incident diagnosis
- **Financial Operations (FinOps)**: 15 scenarios with synthetic cost anomaly data for spend analysis
- **Compliance & Security Operations (CISO)**: 15 scenarios with Kubernetes resources for compliance assessment

## Summary

ITBench evaluates LLMs against reasoning and decision-making for IT automation tasks across three domains:

1. **Site Reliability Engineering (SRE)**: Ensures app resilience and performance through incident diagnosis
2. **Financial Operations (FinOps)**: Manages IT spend by performing cost anomaly analysis
3. **Compliance & Security Operations (CISO)**: Manages threats and assesses policies by conducting compliance posture assessment *(coming soon)*

This leaderboard represents a subset of the broader ITBench and includes selected SRE and FinOps scenarios in offline environments. We will continue enriching the scenario set and incorporating agentic evaluations into the leaderboard over time.

## Key Resources

- [Paper](https://arxiv.org/pdf/2502.05352)
- [GitHub](https://github.com/itbench-hub/ITBench)
- [Kaggle Leaderboard](https://www.kaggle.com/competitions/itbench)

## Dataset Structure

### Directory Organization

```
ITBench-Lite/
├── snapshots/
│   ├── sre/
│   │   └── v0.2-B96DF826-4BB2-4B62-97AB-6D84254C53D7/
│   │       ├── Scenario-1/
│   │       │   ├── alerts/                     # Alert snapshots over time
│   │       │   ├── metrics/                    # Prometheus metrics
│   │       │   ├── ground_truth.yaml          # Ground truth entities
│   │       │   ├── k8s_events_raw.tsv         # Kubernetes events
│   │       │   ├── k8s_objects_raw.tsv        # Kubernetes resources
│   │       │   ├── otel_logs_raw.tsv          # OpenTelemetry logs
│   │       │   └── otel_traces_raw.tsv        # OpenTelemetry traces
│   │       ├── Scenario-2/
│   │       └── ... (35 scenarios total)
│   │
│   ├── finops/
│   │   └── v0.1-finops-anomaly/
│   │       ├── Scenario-1/
│   │       │   └── anomaly.json               # Cost anomaly details
│   │       ├── Scenario-2/
│   │       └── ... (15 scenarios total)
│   │
│   └── ciso/
│       └── v0.1/
│           ├── k8s-opa-static-cis-5.1.1/
│           │   ├── ground-truth/
│           │   │   ├── fetch.sh               # Script to fetch resources
│           │   │   └── policy.rego            # OPA policy definition
│           │   └── static-resources-compliant/ # Kubernetes resources
│           ├── k8s-opa-static-cis-5.1.6/
│           └── ... (15 scenarios total)
```

### Data Files by Domain

#### SRE Scenarios (35 scenarios)

Each SRE scenario contains observability data from orchestrated environments where faults were injected:

| File | Format | Description |
|------|--------|-------------|
| `alerts/` | JSON | Time-series alert snapshots from monitoring systems |
| `metrics/` | Prometheus | Infrastructure and application metrics |
| `ground_truth.yaml` | YAML | Ground truth faulty entities and metadata |
| `k8s_events_raw.tsv` | TSV | Kubernetes cluster events |
| `k8s_objects_raw.tsv` | TSV | Kubernetes resource states (pods, services, deployments) |
| `otel_logs_raw.tsv` | TSV | Application and system logs via OpenTelemetry |
| `otel_traces_raw.tsv` | TSV | Distributed traces via OpenTelemetry |

#### FinOps Scenarios (15 scenarios)

Each FinOps scenario contains synthetic cost anomaly data:

| File | Format | Description |
|------|--------|-------------|
| `anomaly.json` | JSON | Cost anomaly details (date, account_id) |

**Example `anomaly.json`:**
```json
{
    "date": "2025-10-13",
    "account_id": "49tAGZz"
}
```
#### CISO Scenarios (15 scenarios)

Each CISO scenario contains Kubernetes resources and compliance assessment tasks:

| File | Format | Description |
|------|--------|-------------|
| `problem.md` | Markdown | Task description for compliance assessment for agents |
| `static-resources/` | YAML | Kubernetes resources with embedded compliance violations |
| `ground-truth/policy.rego` | Rego | Reference OPA policy solution |
| `ground-truth/fetch.sh` | Shell | Reference resource collection script |
| `static-resources-compliant/` | YAML | Compliant Kubernetes resources (used only for evaluation) |

**Note**: `problem.md` and `static-resources/` are provided as input to agents for solving the compliance assessment task.

## Tasks

### SRE: Fault Localization

Identify the faulty entity or resource (e.g., a specific pod, service, or deployment) that caused the incident based on logs, traces, metrics, Kubernetes events and resources from the environment snapshot.

### FinOps: Cost Anomaly Analysis

Identify the resource responsible for anomalous cost changes using synthetic cost anomaly details and workload configuration.

### CISO: Compliance Assessment

Collect kubernetes resources and generate the correct OPA policy based on regulatory rules and natural-language Kubernetes configurations.

## Limitations

The SRE scenarios are **exported snapshots from sandboxed live Kubernetes environments**. While sourced from real running systems, the snapshot format inherently lacks the dynamic characteristics of live environments:

- Real-time observability (streaming metrics, logs, traces)
- Runtime state changes and non-deterministic behavior
- Interactive debugging and human-in-the-loop investigation
- Active remediation capabilities (deployments, rollbacks, scaling)

The static nature of these exports makes them suitable for diagnostic analysis but does not capture the full complexity of live incident response.

## Related Datasets

- **[ITBench-Trajectories](https://huggingface.co/datasets/ibm-research/ITBench-Trajectories)**: Complete agent execution traces with reasoning steps, tool usage, and evaluation metrics for 105 trajectories across 35 SRE scenarios

## Citation

If you use ITBench in academic or industrial work, please cite:

```bibtex
@misc{jha2025itbench,
  title={ITBench: Evaluating AI Agents across Diverse Real-World IT Automation Tasks},
  author={Jha, Saurabh and Arora, Rohan and Watanabe, Yuji and others},
  year={2025},
  eprint={2502.05352},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2502.05352}
}
```

## Contact & Support

- **GitHub Issues**: [ITBench Issues](https://github.com/itbench-hub/ITBench/issues)
- **Kaggle Discussion**: [ITBench Competition](https://www.kaggle.com/competitions/itbench/discussion)
- **Paper**: [arXiv:2502.05352](https://arxiv.org/abs/2502.05352)

**Last Updated**: January 2026
