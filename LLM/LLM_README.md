# Medical Report Parser with vLLM

A configurable pipeline for extracting structured data from medical reports using DeepSeek LLMs.

## Key Features

- **Structured Information Extraction**: Converts unstructured medical reports into structured data.
- **Multi-Format Support**: Handles CSV, JSON, and raw text inputs.
- **Customizable Extraction**: Define fields and validation rules via YAML.
- **Automated Quality Control**: Ensures data completeness and consistency.
- **Adaptive Retries**: Handles missing or incomplete extractions with intelligent retries.

---

## Installation

```bash
# Create environment
python -m venv MedicalLLM
source MedicalLLM/bin/activate

# Install dependencies
pip install vllm pandas pyyaml

# Clone repository
git clone https://github.com/your/repo.git
cd LLM
```

---

## How to Use

The parser can be executed using the provided `run.py` script. Below is an example usage:

```bash
python run.py \
  --input path/to/input.csv \
  --output path/to/output.csv \
  --config configs/pathology.yaml \
  --model V3 \
  --format csv
```

### Computation power

Each DeepSeek model has different GPU requirements:

- **DeepSeek-V3** and **DeepSeek-R1**: Running these models effectively requires a substantial amount of GPU memory. The dequantized 16-bit version takes up over a terabyte of storage and would likely need over 800GB of RAM/VRAM.  However, it is possible to run the model with at least 180GB of combined VRAM and RAM, on their designed precision.

- **DeepSeek-R1-Distill-Qwen-32B**: This model is about 20x smaller than the larger DeepSeek-V3 and R1. I managed to get it working on 2 A40 cards, so it still needs quite a lot of compute.

### Command-Line Arguments

| Argument | Description |
|----------|-------------|
| `-i, --input` | Path to input file (CSV or JSON) |
| `-o, --output` | Path to save the processed output |
| `-c, --config` | Path to YAML configuration file |
| `-m, --model` | Model to use (`V3`, `R1`, `R1-Distill`) |
| `-f, --format` | Input file format (`csv` or `json`) |
| `--text-key` | Key for text in JSON (default: `text`) |
| `--report-type-col` | Column name for report type in CSV (default: `reportType`) |
| `--text-col` | Column name for text in CSV (default: `presentedForm_data`) |
| `-g, --gpus` | Number of GPUs to use (default: 2) |
| `-a, --attempts` | Max LLM retry attempts (default: 3) |
| `-r, --regex` | Path to JSON file with regex patterns for extraction |

---

## Defining Extraction Rules with YAML

The pipeline is configured via a YAML file that defines the whole prompt, including system instructions, which fields to extract and how to validate them, and optionally an example case. In resource you can find an example YAML file.

### Example YAML Configuration

```yaml
report_type: "Pathology"
system_instruction: |
  [SYSTEM] You are a medical data extraction system...

field_instructions:
  - name: "Specimen Type"
    type: "string"
    options: ["Biopsy", "Resection"]
    default: "Not specified"

task: |
  [TASK] Extract into this structure:
  ```json
  {
    "Field1": ""
  }```
```

### Supported Field Types

| Type    | Example Configuration |
|---------|------------------------|
| **Simple** | `type: "string"` or `type: "number"` |
| **Options** | `options: ["Yes", "No"]` |
| **Nested** | ` type: "nested"`<br> `key: "subfield"` |
| **List** | `type: "list"`<br> `item_type: "dict"` |

---

## Core Components

### VLLMReportParser (`parser.py`)

The main component responsible for parsing medical reports.

| Method | Description |
|--------|------------|
| `__init__()` | Initializes LLM with the specified model and config. |
| `process_texts()` | Converts text reports into structured data. |
| `_generate_query()` | Constructs prompts for the LLM. |
| `_parse_response()` | Extracts and validates structured output from the LLM response. |
| `find_missing_fields()` | Identifies incomplete extractions and triggers follow-ups. |
| `process_with_adapter()` | Integrates the parser with different input formats. |

### Adapters (`adapters/`)

Adapters handle input formats and structure parsed outputs accordingly.

| Adapter | Best For | Key Methods |
|---------|----------|-------------|
| `DataFrameAdapter` | CSV/Excel files | `prepare_inputs()`, `merge_results()` |
| `JsonAdapter` | API/JSON responses | `prepare_inputs()`, `format_outputs()` |

#### Example: Using a CSV Adapter
```python
DataFrameAdapter(
    df=pd.read_csv("input.csv"),
    text_column="report_text"
)
```

#### Example: Using a JSON Adapter
```python
JsonAdapter(
    input_path="data.json",
    text_key="content"
)
```

### Creating a Custom Adapter
```python
from adapters.base_adapter import BaseAdapter

class CustomAdapter(BaseAdapter):
    def prepare_inputs(self) -> List[str]:
        # Custom logic for processing input
        return processed_texts
        
    def format_outputs(self, results: List[Dict]) -> Any:
        # Custom logic for formatting output
        return formatted_output
```

---

## Running the Parser

```python
from parser import VLLMReportParser
from adapters import DataFrameAdapter

parser = VLLMReportParser(
    model="deepseek-ai/DeepSeek-V3",
    yaml_config="configs/pathology.yaml"
)

adapter = DataFrameAdapter(
    df=pd.read_csv("reports.csv"),
    text_column="text_content"
)

results = parser.process_with_adapter(adapter)
results.to_csv("output.csv")
```

---