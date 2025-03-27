# MedicalRecordNLP

## LLM

A configurable pipeline for extracting structured data from medical reports using DeepSeek LLMs. It converts unstructured reports into structured data and supports multiple formats, with customizable extraction rules and automated quality control. The pipeline includes adaptive retries for missing data and can be easily integrated with CSV, JSON, and raw text inputs.

For detailed instructions, please refer to the [LLM README](LLM/LLM_README.md).

## NLP

This NLP pipeline extracts entities from semi-structured reports using techniques like translation, Named Entity Recognition (NER), Entity Linking, and dependency analysis. It includes specialized scripts for pathology reports, with a focus on extracting tumor-related information. The pipeline also incorporates handcrafted matchers and regex patterns for precise extraction.

For detailed instructions, please refer to the [NLP README](NLP/NLP_README.md).