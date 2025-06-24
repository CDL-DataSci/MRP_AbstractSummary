# MRP_AbstractSummary
Protecting PII when generating Abstract Summary text from datasource.

# Introduction

Access to public records is often hindered by administrative, legal, and technical challenges. Key barriers include the labor-intensive process of reviewing documents against privacy and access laws, difficulties in identifying relevant records using non-standardized content, and the high cost of processing large volumes of documents. This research proposes leveraging Large Language Models (LLMs) and automatic text summarization to address these challenges and improve access to public records.
The current manual processes for identifying, reviewing, and retrieving public records are time-consuming, costly, and often fail to efficiently address requests due to:
* The need to assess compliance with complex legal requirements (e.g., Privacy Act, PIPEDA, FIPPA, MFIPPA).
* Limitations of keyword-based search in identifying responsive documents.
* High resource cost associated with reviewing large volumes of documents or lengthy reports.
  
This literature review will explore current research in using traditional pre-trained transformer models specialized in named entity recognition (NER) and LLMs such as LLaMA 3 to correctly identify personally identifiable information (PII), ultimately seeking to prevent leaking PII from the fine-tuned model training set, and to finally produce document summaries that are clear and coherent and no do not contain PII.

# Contents

This repo contains the following:
* Collection of articles that make up the Literature Review
* Full dataset of all council and committee meeting minutes (2008 to 2025)

The repo will be kept up to date as the MRP progresses.

