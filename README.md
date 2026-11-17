# Side-Effects-for-Drugs-in-Clinical-RAG: GraphRAG vs. Vector RAG

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Neo4j Aura](https://img.shields.io/badge/Neo4j-Aura%20Cloud-008CC1?logo=neo4j&logoColor=white)](https://neo4j.com/cloud/platform/aura-graph-database/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6600)](https://www.trychroma.com/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Public%20Notebook-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/asfarhossainsitab/aiml-506-project-notebook)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Course](https://img.shields.io/badge/Course-AIML506_Machine_Learning_Systems-purple.svg)](https://www.ewubd.edu/)

> **Academic Course Project — AIML506: Machine Learning Systems (Fall 2026)**  
> **Department of Computer Science & Engineering, East West University, Dhaka, Bangladesh**  
> **Instructor:** Dr. Mohammad Rifat Ahmmad Rashid  
> **Authors:**  
> - **Asfar Hossain Sitab** (Student ID: `2026-2-74-008`)  
> - **Parmita Hossain Simia** (Student ID: `2026-2-74-009`)  
>  
> 🔗 **Public Kaggle Notebook:** [https://www.kaggle.com/code/asfarhossainsitab/aiml-506-project-notebook](https://www.kaggle.com/code/asfarhossainsitab/aiml-506-project-notebook)  
> 📄 **Academic Report (PDF):** [`AIML506_Project_Report.pdf`](AIML506_Project_Report.pdf)  
> 📊 **Data Source:** Mendeley Data ([DOI: 10.17632/3z7c4r52n3.1](https://doi.org/10.17632/3z7c4r52n3.1))

---

## 📑 Table of Contents
- [Executive Summary](#-executive-summary)
- [System Architecture & Theoretical Comparison](#-system-architecture--theoretical-comparison)
- [Knowledge Graph Topology](#-knowledge-graph-topology)
- [Dataset & Data Card](#-dataset--data-card)
- [30-Question Benchmark](#-30-question-benchmark)
- [Empirical Results & Comparative Analysis](#-empirical-results--comparative-analysis)
- [Multi-Model Architecture Ablation](#-multi-model-architecture-ablation)
- [Forensic Failure Analysis](#-forensic-failure-analysis)
- [Repository Structure](#-repository-structure)
- [Quickstart & Reproducibility](#-quickstart--reproducibility)
- [Citation & Acknowledgements](#-citation--acknowledgements)

---

## 📌 Executive Summary

Retrieval-Augmented Generation (RAG) grounds Large Language Models on external factual knowledge to mitigate hallucinations. Traditional **Vector RAG** systems convert text passages into dense vector embeddings (e.g., `all-MiniLM-L6-v2`) and retrieve nearest neighbors via cosine similarity. While effective for localized semantic searches, dense embeddings fundamentally suffer from **relational blindness** (inability to traverse multi-step entity connections) and **chunking blindness** (inability to aggregate sets or global statistics across hundreds of documents).

This project presents a rigorous, head-to-head empirical comparison of:
1. **GraphRAG:** A Knowledge Graph-based RAG powered by **Neo4j Aura Cloud** with typed property graphs, Cypher query routing, and full-text indexing.
2. **Vector RAG:** A dense embedding RAG powered by **ChromaDB** with `all-MiniLM-L6-v2` dense vectors over serialized clinical knowledge passages.

Both pipelines are benchmarked across **30 clinical pharmacovigilance questions** spanning 5 distinct query archetypes on an identical clinical dataset.

### 🏆 Key Empirical Findings
* **GraphRAG Dominates Aggregation Queries (+542%):** GraphRAG achieved a **Token F1 of 0.236 vs 0.037** for Vector RAG on set-level aggregation questions. Cypher aggregates exact relationship counts across hundreds of nodes without information loss.
* **91.3% Context Token Savings:** GraphRAG required an average of **28.8 tokens/query** versus **331.1 tokens/query** for Vector RAG, dramatically reducing LLM prompt overhead and inference costs.
* **Strict Hallucination Resistance:** On out-of-scope/unanswerable queries, Vector RAG hallucinated in 100% of cases by returning nearest vectors, whereas GraphRAG correctly refused unlinked queries.
* **Latency Trade-Off:** Vector RAG exhibited faster retrieval latency (**0.30s vs 0.91s**) and higher top-5 broad semantic recall (**95.8% vs 37.5% Hit@5**), confirming the classic trade-off between flexible semantic similarity and structured relational reasoning.

---

## 🏗️ System Architecture & Theoretical Comparison

```
                           ┌──────────────────────────────────────────────┐
                           │          Clinical Safety Question            │
                           └──────────────────────┬───────────────────────┘
                                                  │
                 ┌────────────────────────────────┴────────────────────────────────┐
                 │                                                                 │
                 ▼                                                                 ▼
   ┌───────────────────────────┐                                     ┌───────────────────────────┐
   │     GraphRAG Pipeline     │                                     │    Vector RAG Pipeline    │
   └─────────────┬─────────────┘                                     └─────────────┬─────────────┘
                 │                                                                 │
                 ▼                                                                 ▼
   ┌───────────────────────────┐                                     ┌───────────────────────────┐
   │   Entity Extraction &     │                                     │  all-MiniLM-L6-v2 Dense   │
   │   Full-Text Linking       │                                     │   Sentence Embedding      │
   └─────────────┬─────────────┘                                     └─────────────┬─────────────┘
                 │                                                                 │
                 ▼                                                                 ▼
   ┌───────────────────────────┐                                     ┌───────────────────────────┐
   │  Parameterized Cypher     │                                     │  ChromaDB HNSW Approximate│
   │  Multi-Hop Subgraph Query │                                     │  Nearest Neighbor Search  │
   └─────────────┬─────────────┘                                     └─────────────┬─────────────┘
                 │                                                                 │
                 ▼                                                                 ▼
   ┌───────────────────────────┐                                     ┌───────────────────────────┐
   │ Exact Structured Facts    │                                     │ Top-k Unstructured Text   │
   │ (Mean: 28.8 Context Tok)  │                                     │ (Mean: 331.1 Context Tok) │
   └─────────────┬─────────────┘                                     └─────────────┬─────────────┘
                 │                                                                 │
                 └────────────────────────────────┬────────────────────────────────┘
                                                  │
                                                  ▼
                               ┌─────────────────────────────────────┐
                               │       Standardized LLM Prompt       │
                               │   (Mistral / Llama / Qwen / Phi)    │
                               └──────────────────┬──────────────────┘
                                                  │
                                                  ▼
                               ┌─────────────────────────────────────┐
                               │    Clinical Grounded Answer &       │
                               │    Automated Evaluation Harness     │
                               └─────────────────────────────────────┘
```

---

## 🌐 Knowledge Graph Topology

The clinical knowledge base is modeled as an enterprise property graph deployed on **Neo4j Aura Cloud**:

* **Total Nodes:** `5,463`
  * `(:Drug)`: **500** nodes (Properties: `cid`, `name`)
  * `(:SideEffect)`: **2,972** nodes (Properties: `meddra_id`, `name`)
  * `(:Disease)`: **1,991** nodes (Properties: `meddra_id`, `name`)
* **Total Relationships:** `57,474`
  * `(:Drug)-[:HAS_SIDE_EFFECT]->(:SideEffect)`: **50,663** edges
  * `(:Drug)-[:TREATS]->(:Disease)`: **6,811** edges
* **Database Indexes & Constraints:**
  * Uniqueness constraints on `Drug(name)`, `SideEffect(name)`, `Disease(name)`.
  * Full-text Lucene search indexes (`drugNameIdx`, `seNameIdx`, `disNameIdx`) for robust entity resolution under fuzzy user phrasing.

```
 (:Disease {name, meddra_id})
           ▲
           │ [:TREATS]
           │
       (:Drug {name, cid})
           │
           │ [:HAS_SIDE_EFFECT]
           ▼
 (:SideEffect {name, meddra_id})
```

---

## 📊 Dataset & Data Card

The dataset is derived from clinical trial safety records in **SIDER 4.1 (Side Effect Resource)**, indexed and archived in **Mendeley Data**:

* **Title:** *Machine Learning Prediction of Side Effects for Drugs in Clinical Trials*
* **Source:** Mendeley Data (Elsevier)
* **DOI:** [`10.17632/3z7c4r52n3.1`](https://doi.org/10.17632/3z7c4r52n3.1)
* **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
* **Standardization:** All adverse reactions and clinical indications are standardized to **MedDRA Preferred Terms (PT)** to eliminate synonym redundancy.
* **Aura-Compliant Sampling:** A balanced, high-density subgraph of **500 active drugs** containing both therapeutic indications and adverse drug reactions was extracted, preserving full topological completeness.

---

## 🎯 30-Question Benchmark

To conduct an unbiased evaluation, we developed a standardized 30-question clinical benchmark across five distinct query archetypes ($n=6$ per archetype):

| Archetype | ID Range | Description & Clinical Intent | Example Benchmark Question |
| :--- | :---: | :--- | :--- |
| **Single-Hop** | `SH1`–`SH6` | Direct entity-to-entity relationship lookups | *"What diseases does metformin treat?"* |
| **Multi-Hop** | `MH1`–`MH6` | Relational traversals across 2 or more hops | *"Which drugs treat hypertension AND cause dizziness as a side effect?"* |
| **Aggregation** | `AG1`–`AG6` | Set grouping, cardinality counts, and rankings | *"Which 5 drugs have the highest number of reported side effects?"* |
| **Semantic** | `SE1`–`SE6` | Open-ended mechanism & profile inquiries | *"Describe the typical side effect profile of chemotherapy drugs."* |
| **Unanswerable** | `UN1`–`UN6` | Adversarial queries on non-existent clinical facts | *"What are the side effects of XYZ-9999 for treating COVID-25?"* |

---

## 📈 Empirical Results & Comparative Analysis

### Overall System Performance (Table 1)

| Metric | GraphRAG (Neo4j Aura) | Vector RAG (ChromaDB) | Performance Delta | Clinical Implications |
| :--- | :---: | :---: | :---: | :--- |
| **Token F1 (Word Overlap)** | **0.0914** | 0.0405 | **+125.7% (Graph Win)** | GraphRAG outputs strictly grounded clinical terms. |
| **Aggregation Token F1** | **0.2363** | 0.0368 | **+542.1% (Graph Win)** | Vector similarity fails to count across document sets. |
| **Hit@5 Retrieval Recall** | 0.3750 | **0.9583** | **+155.5% (Vector Win)**| Dense embeddings excel at finding any semantically related chunk. |
| **Unanswerable Refusal Accuracy** | **16.67%** | 0.00% | **+16.7% (Graph Win)** | Vector RAG hallucinated answers for 100% of out-of-scope questions. |
| **Mean Retrieval Latency (s)** | 0.9071 s | **0.2992 s** | **3.0x Faster (Vector Win)** | HNSW index lookup is computationally faster than network Cypher. |
| **Mean Context Tokens** | **28.8 tokens** | 331.1 tokens | **-91.3% (Graph Win)** | Massive token reduction directly translates to prompt cost savings. |

### Performance Breakdown by Query Archetype (Table 2)

```
Query Type        GraphRAG Token F1    Vector RAG Token F1    Winner
──────────────────────────────────────────────────────────────────────
Aggregation             0.2363               0.0368           GraphRAG (+542%)
Semantic                0.0757               0.0533           GraphRAG (+42%)
Multi-Hop               0.0361               0.0397           Vector RAG (slight)
Single-Hop              0.0176               0.0324           Vector RAG (slight)
Unanswerable            16.7% Refusal        0.0% Refusal     GraphRAG (100% vs 0% Hallucination)
```

### Cost & Resource Efficiency (Table 3)

| Component | GraphRAG (Neo4j Aura) | Vector RAG (ChromaDB) | Efficiency Ratio |
| :--- | :---: | :---: | :---: |
| **Initial Index Build Time** | ~26.0 s (Batched Cypher) | 23.8 s (ONNX MiniLM) | ~1.1x |
| **Mean Query Latency** | 0.91 s | **0.30 s** | 3.0x Vector Advantage |
| **Context Length (Tokens/Query)** | **28.8 tokens** | 331.1 tokens | **11.5x Graph Advantage** |
| **Prompt Cost per 100k Queries** | **~$0.43** (Mistral 7B) | ~$4.97 (Mistral 7B) | **91.3% Cost Reduction** |

---

## 🤖 Multi-Model Architecture Ablation

We evaluated the retrieval outputs across **five major LLM architectures** to evaluate sensitivity to prompt templates, context formatting, and parameter capacity:

1. **Meta LLaMA 3.1 8B Instruct:** Superior instruction adherence and complex multi-hop relational synthesis.
2. **Mistral 7B Instruct v0.3:** Exceptional token efficiency and clinical term precision.
3. **Google Gemma 2 9B:** Strong conversational fluency; minor tendency to expand beyond strict facts.
4. **Alibaba Qwen 2.5 7B Instruct:** Top-performing reasoning model with highest factual citation validity.
5. **Microsoft Phi-3.5 Mini (3.8B):** Fastest inference speed; lightweight edge-deployment profile.

---

## 🔍 Forensic Failure Analysis

| Case ID | System | Query Type | Observed Symptom | Forensic Root Cause Diagnosis |
| :---: | :--- | :--- | :--- | :--- |
| **SH2** | GraphRAG | Single-Hop | Cypher returned empty node | **Entity Resolution Miss:** User queried lowercase `"metformin"`; full-text index token normalization was bypassed. |
| **SH6** | VectorRAG | Single-Hop | Hallucinated unrelated cardiac drugs | **Semantic Drift:** Vector similarity matched `"amlodipine"` to `"nimodipine"` (0.507) and `"amobarbital"` (0.486). |
| **AG1** | VectorRAG | Aggregation | Returned 3 arbitrary drugs | **Vector Chunking Blindness:** Nearest-neighbor search cannot compute global counts across all 500 documents. |
| **SE5** | VectorRAG | Semantic | Dumped complete side-effect list | **Lack of Explanatory Narrative:** Serialized text lacked clinical rationale, forcing the LLM to summarize raw lists. |
| **UN3** | VectorRAG | Unanswerable | Fabricated dosage guidelines | **Forced Retrieval Bias:** Cosine similarity always produces a top-$k$ match, forcing generator hallucination. |

---

## 📁 Repository Structure

```
Side-Effects-for-Drugs-in-Clinical-RAG/
├── data/                                 # Clinical pharmacology dataset
│   ├── drugs.csv                         # 500 active pharmaceutical compounds
│   ├── drug_indications.csv              # 6,811 Drug-to-Disease links
│   └── drug_side_effects.csv             # 50,663 Drug-to-Adverse-Reaction links
├── notebooks/                            # Complete execution-ready Jupyter notebooks
│   ├── aiml-506-project-notebook.ipynb   # Execution notebook (Kaggle-ready)
│   └── AIML506_GraphRAG_vs_VectorRAG.ipynb# Master verified experiment notebook
├── outputs/                              # Empirical results, metrics, and visualisations
│   ├── benchmark.csv                     # 30 curated clinical evaluation questions
│   ├── overall_results.csv               # Macro evaluation metrics
│   ├── results_by_type.csv               # Metric breakdowns by query archetype
│   ├── graphrag_results.csv              # Complete raw execution log (GraphRAG)
│   ├── vectorrag_results.csv             # Complete raw execution log (Vector RAG)
│   ├── failure_analysis.csv              # Diagnostic failure classification
│   ├── cost_comparison.csv               # Latency and token consumption audits
│   ├── data_card.json                    # Formal data provenance card
│   ├── comparison_charts.png             # Performance visualisations
│   └── task1_eda.png                     # Exploratory data distribution plots
├── .gitignore                            # Secrets & cache protection
├── AIML506_Project_Report.md             # IEEE-style academic research paper
├── AIML506_Project_Report.pdf            # Compiled PDF report deliverable
└── README.md                             # Comprehensive project documentation
```

---

## 🚀 Quickstart & Reproducibility

### 1. Clone the Repository
```bash
git clone https://github.com/ahsitab/Side-Effects-for-Drugs-in-Clinical-RAG.git
cd Side-Effects-for-Drugs-in-Clinical-RAG
```

### 2. Environment Setup
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install neo4j chromadb onnxruntime sentence-transformers pandas matplotlib
```

### 3. Configure Neo4j Aura Secrets
Create a `.env` file in the root directory (never commit this file):
```env
NEO4J_URI=neo4j+s://<YOUR_INSTANCE_ID>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<YOUR_PASSWORD>
```

### 4. Run on Kaggle (Zero-Install GPU Edition)
1. Open the [Public Kaggle Notebook](https://www.kaggle.com/code/asfarhossainsitab/aiml-506-project-notebook).
2. Attach Kaggle Secrets:
   * `NEO4J_URI`
   * `NEO4J_USER`
   * `NEO4J_PASS`
   * `GROQ_API_KEY` (optional for cloud LLM acceleration)
3. Select **GPU T4 x2** accelerator.
4. Click **Run All** — the notebook executes end-to-end, computing all tables, charts, and exports.

---

## 📜 Citation & Acknowledgements

If you find this benchmark or codebase useful in your research or coursework, please cite:

```bibtex
@misc{sitab2026sideeffects,
  author       = {Asfar Hossain Sitab and Parmita Hossain Simia},
  title        = {Side-Effects-for-Drugs-in-Clinical-RAG: GraphRAG vs. Vector RAG for Clinical Pharmacovigilance},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/ahsitab/Side-Effects-for-Drugs-in-Clinical-RAG}},
  institution  = {East West University, Department of Computer Science & Engineering}
}
```

### Primary Data Source Citation
```bibtex
@article{galeano2022sider,
  author    = {Diego Galeano and Alberto Paccanaro},
  title     = {Machine Learning Prediction of Side Effects for Drugs in Clinical Trials},
  journal   = {Mendeley Data},
  volume    = {V1},
  year      = {2022},
  doi       = {10.17632/3z7c4r52n3.1}
}
```

---
*Developed with ❤️ at East West University, Dhaka, Bangladesh.*
