# Understanding the Impact of Coreference Resolution and Structured Prompts in LINK-KG​

## Overview

This project studies the impact of two key components in the LINK-KG framework for constructing knowledge graphs from unstructured legal documents related to human smuggling:
	1.	Cache-based coreference resolution
	2.	Structured prompting for entity and relationship extraction

The work is divided into two main parts:

1. Ablation Study on Knowledge Graph Quality

We perform a controlled ablation study by isolating the two core components of LINK-KG to understand their individual contributions to knowledge graph construction. Specifically, we compare four pipelines:
	•	Full LINK-KG (with coreference resolution + structured prompts)
	•	LINK-KG-no-coref (without coreference resolution)
	•	LINK-KG-no-str-prompt (without modified structured prompts)
	•	GraphRAG baseline (without both components)

Using 16 legal case documents, we evaluate how removing each component affects:
	•	Node duplication
	•	Legal noise
	•	Overall structural quality of the generated knowledge graphs

This allows us to quantify how coreference resolution and structured prompting independently and jointly influence knowledge graph quality.

2. Entity and Relationship (ER) Extraction Evaluation

In parallel, we evaluate the effectiveness of LINK-KG’s structured prompting strategy for entity and relationship extraction using a manually annotated dataset of 541 legal text samples. The evaluation measures:
	•	Global entity and relation extraction performance
	•	Per-entity-type performance across seven legal entity categories

This analysis highlights both the strengths and current limitations of structured prompting for ER extraction in dense legal narratives.

## Instructions to Run the Code

The repository includes the complete implementation of both the LINK-KG pipeline and its ablation variants (LinkKG-no-coref, LinkKG-no-str-prompt, and GraphRAG).

### Requirements
- Python: 3.12
- Ollama (for local LLM inference): https://ollama.com/download
- GraphRAG version 0.3.2 (for KG construction baseline): https://github.com/microsoft/graphrag
  
### Dataset Links
- Ablation Study Cases (16 Legal Documents): 
  https://github.com/dipakmeher/cs782AdvancedML/tree/main/linkkg-no-str-prompt/input
  
- NER Evaluation Dataset (Annotated CSV):  
  https://github.com/dipakmeher/cs782AdvancedML/blob/main/legal_ner.csv
   
### Commands

#### 1. Run Coreference Resolution (LINK-KG)

Use the following command template to run the coreference resolution pipeline. This resolves entity mentions in a legal document using a type-specific LLM prompt.

```bash
python run_pipeline5.py \
  --input-file <path_to_input_text> \
  --input-file-name <file_id_without_extension> \
  --entity-type <entity_type> \
  --max-tokens 300 \
  --min-last-chunk-words 50 \
  --use-tokenizer \
  --ner-prompt-file <path_to_ner_prompt> \
  --ner-model-name <model_name_in_ollama> \
  --coref-prompt-file <path_to_coref_prompt> \
  --coref-model-name <model_name_in_ollama> \
  --resolve-prompt-file <path_to_resolve_prompt> \
  --resolve-model-name <model_name_in_ollama> \
  --run-stages chunk ner coref resolve
```

##### Argument Descriptions
- `--input-file`: Path to the raw document (e.g., `.txt` file)
- `--input-file-name`: Unique name used to create output folders
- `--entity-type`: One of person, location, routes, etc.
- `--run-stages`: Sequence of pipeline stages to run (chunk, ner, coref, resolve)

#### LINK-KG Prompt Files

- Coreference Prompts: `link-kg/prompts/`
- KG Construction Prompts: `link-kg/kgconstruction/ragtest/prompts/`

#### 2. Run KG Construction (GraphRAG-based)

```bash
python index.py --root ./ragtest
```

Make sure you are in the correct directory and have installed GraphRAG dependencies:  
https://github.com/microsoft/graphrag


#### 3. Run KG Construction (For all 4 ablation variants)

```bash
python index.py --root ./ragtest
```

Ensure directory structure is consistent with GraphRAG format.

#### 4. Baseline: GraphRAG Only

To run GraphRAG as a standalone baseline:

```bash
python index.py --root ./ragtest
```

- Prompts: Baseline prompt templates are located in `baseline/ragtest/prompts/`

