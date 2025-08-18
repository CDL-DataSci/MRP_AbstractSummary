# MRP_AbstractSummary
Protecting PII when generating Abstract Summary text from datasource.

# Protecting Personal Information When Finetuning LLM

The aim of this study is to evaluate the effectiveness of a fine-tuned LLaMA 3 Retrieval-Augmented Generation (RAG) model for automatically generating coherent, relevant, and privacy-preserving summaries of the Town of Whitby's council and committee meeting minutes. 

Public access to government records is often restricted by manual and resource-intensive review processes to identify relevant records and then redact content to prevent the exposure of personally identifiable information (PII). Leveraging recent advancements in machine learning and large language models, particularly methods such as Differential Private Data Dropout (DPDD), Named Entity Recognition (NER), Retrieval-Augmented Generation (RAG), this research seeks to evaluate whether these innovative approaches can simultaneously improve access while protecting privacy. As part of the assessment pf privacy protection, this research incorporates Canary Extraction Success Rate (CESR) which is a targeted measure of memorization risk that involves the strategic insertion of identifiable false statements ("canaries") that will allow for the evaluation of privacy leakage risks from the large language model used.

However, as the practical deployment of large language models is constrained by computational resources, this study includes Low-Rank Adaptation (LoRA) as a method to mitigate computational costs and to provide the flexibility to scale the model as required.

The current manual processes for identifying, reviewing, and retrieving public records are time-consuming, costly, and often fail to efficiently address requests due to:
* The need to assess compliance with complex legal requirements (e.g., Privacy Act, PIPEDA, FIPPA, MFIPPA).
* Limitations of keyword-based search in identifying responsive documents.
* High resource cost associated with reviewing large volumes of documents or lengthy reports.

This experiment leverages the Llama 3.1b model, which is available here: https://huggingface.co/meta-llama/Llama-3.1-8B

# Contents

This repo contains the following:
* README.md: File documenting approach to research project and steps to reproduce outcomes.
* LitReview: Collection of articles that make up the Literature Review, and Literature Review Report
* Methodolgies: Submission of Methodolgy & Experiments Report
* Results: Submission of Results Report
* Whitby_Minutes_2025: Full dataset of all council and committee meeting minutes (2008 to 2025)
* WhitbyPII: Directory containing all necessary py scripts and CSV files to reproduce project

The repo will be kept up to date as the MRP progresses.

# Experimental Design:
The experimental design incorporates a factorial structure varying LoRA rank, retrieval settings, and DPDD preprocessing. Additionally, stratified sampling based on document length categories and publication years (grouped by periods identified during exploratory data analysis) will ensure robust performance across subsets of data. Data will be partitioned into distinct train-validation-test splits (70%-15%-15%), ensuring that hyperparameter tuning is performed using validation sets, while final model evaluation occurs exclusively on the test set.

Potential limitations include variability introduced by noisy or anomalous data, inherent randomness in model initialization, and complexities arising from interactions among multiple factors.

Factor Design:
•	Conduct experiments in a structured way, varying multiple factors simultaneously (LoRA rank, Retrieval Documents, and DPDD settings).
•	Train-validation-test splits for evaluation of model performance (70%-15%-15%)

# Methodology
* Step 1A: Preprocessing & Data Extraction (see step1a.py)
* Step 1B: Defining PII with NER (PERSON, ORG, NORP, FAC) and Calculating Proxy DPDD Scores (see step1b.py)
* Step 1C: TRAIN-TEST-VALIDATION SPLIT (70%-15%-15%) (see step1c.py)
* Step 1D: Inserting CANARIES into TEST.csv (see step1d.py & canaries.csv)
* Step 2: Finetuning Llama-3.1-8B and Factor Experiment (24 iterations of different combinations of factors, 2 replications for each iternation, and 2 epochs each run) (see run_experiments.py)
* Step 3: Repeating the Factor Experiment, but with DPDD filtering to drop high-risk data during training.(see run_experiments_dpdd.py)
* Step 4A: Conduct analysis of evaluation metrics, which includes analysis of training log outputs (validation loss and DPDD scores - see TrainLog_evaluate.py), plus calculting ROUGE, BERTScore of generated summaries from baseline and DPDD-filtered results (see evaluate_metrics.py).
* Step 4B: Conduct ANOVA against experiment factors (see evaluate_ANOVA.py) 
* Step 5: Pass set prompts against basline and DPDD-filtered models to generate new outputs. Conducting Canary Extraction Success Rate (CESR) and Precision and Recall on both the baseline results and DPDD filtered results. (see evaluate_memorization.py)
* Step 6: Human Review of Generated Summaries to compare with CESR results (selection of sample of 50 generated summaries for presence or absence of PII and canaries) (see prompt_probe.csv)


# Exploratory Data Analysis

The Town of Whitby council and committee minute dataset from 2008 to 2025 contains:
* Origianlly contained 1180 individual PDF files, reduced to 1167 after duplications removed
* 10,941 pages in total
* A mean of 9.27 pages per file 
* A median value of 6 pages.
* Total word count of 2,429,861 words across all files.
* A mean value of 2059.20 words per file
* A median value of 1101.5 words per file

## Total Files Per Year
<img width="315" height="235" alt="image" src="https://github.com/user-attachments/assets/96817679-0766-4023-aa9e-f0b9efac8137" />

## Summary Stastics
<img width="432" height="338" alt="image" src="https://github.com/user-attachments/assets/527455ba-6daf-4fd1-84aa-da8f109e7460" />

## File Categories

* Accessibility Advisory Committee
*	Active Transportation and Safe Roads Advisory Committee
*	Animal Services Appeal Committee
*	Audit Committee
* Brooklin Downtown Development Steering Committee
*	Committee of Adjustment
*	Committee of the Whole
*	Compliance Audit Committee
*	Downtown Whitby Development Steering Committee
*	Heritage Whitby Advisory Committee
*	Joint Accessibility Advisory and Diversity and Inclusion Advisory Committees
*	Municipal Licensing and Standards Committee
*	Property Standards Appeal Committee
* Public Meetings
*	Regular Council Meetings
*	Special Council Meetings
*	Whitby Diversity and Inclusion Advisory Committee
*	Whitby Sustainability Advisory Committee

## Histogram: NER Count Across All Files
<img width="468" height="233" alt="image" src="https://github.com/user-attachments/assets/24096561-5d25-4026-8116-2a13c46e3533" />

# Results
## Table 1: Average Proxy DPDD Scores
<img width="432" height="407" alt="image" src="https://github.com/user-attachments/assets/c994b57c-6b69-4c22-92af-fa14939ad8a5" />

## Table 2: Risk Scores Per File
<img width="432" height="284" alt="image" src="https://github.com/user-attachments/assets/75108e4e-566c-4b40-9943-0a3399f768ed" />

## Fig 1: Mean Risk Score by File Category
<img width="364" height="291" alt="image" src="https://github.com/user-attachments/assets/ef2b78cb-4c95-4862-859f-5e4661b9c440" />

## Fig 2: File Count by Risk Score
<img width="432" height="218" alt="image" src="https://github.com/user-attachments/assets/19da1166-9ee3-4893-ab3c-b312c90ead1e" />

Trend showing categories with approximately 100 files or greater tend to have higher proxy risk scores. In this case, the Committee of the Whole, Regular Council Meetings and Committee of Adjustment.

## Table 3: First 5 Runs of the Baselining Factoral Experiment with 24 Combinations.
<img width="432" height="391" alt="image" src="https://github.com/user-attachments/assets/01cf5e93-670d-4cf6-a1b8-d9635627bb68" />

## Fig 3: Ranking Factor Configuration by Lowest Validation Loss.
<img width="468" height="375" alt="image" src="https://github.com/user-attachments/assets/f33eab5b-2c74-4355-a875-e10cce0fe4ba" />

## Table 4: Memorization Probe Summary
Variant	Prompts	Recall	Precision	CESR	Total Canaries	Leak % 
Baseline	25	0.232143	0.565217	0.344262	21	84
DPDD	25	0.089286	0.714286	0.131148	8	32
<img width="432" height="89" alt="image" src="https://github.com/user-attachments/assets/37b7e851-f231-4c95-87b1-64b674cc19b2" />












