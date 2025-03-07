import argparse
from pathlib import Path
import os
import json
import pandas as pd
import re
import warnings
import logging
from thefuzz import fuzz
from typing import Optional, Dict

from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def compile_patterns(patterns: Dict) -> Dict:
    """
    Compiles regular expression patterns from a dictionary of pattern configurations.

    Parameters
    ----------
    patterns : dict
        A dictionary where each key is a pattern name, and the value is another dictionary
        containing the 'pattern' (a regular expression string), 'flags' (string representing
        regex flags), and 'start' (the starting point of the pattern in the text).

    Returns
    -------
    dict
        A dictionary where each key corresponds to the pattern name, and the value is a dictionary
        with the compiled regex object ('pattern') and the starting point ('start').
    """
    compiled = {}
    for name, pat in patterns.items():
        flags = getattr(re, pat["flags"], 0)
        compiled[name] = {
            "pattern": re.compile(pat["pattern"], flags),
            "start": pat["start"],
        }
    return compiled

def extract_text(translation: str, regex: Dict) -> str:
    """
    Extracts text from a translation based on regular expression patterns.

    Parameters
    ----------
    translation : str
        The translated pathology report.
    regex : dict
        A dictionary containing the compiled regular expression patterns.

    Returns
    -------
    str
        The extracted text from the translation based on the patterns.
    """
    extracted = []
    for pattern in regex.values():
        match = pattern["pattern"].search(translation)
        if match:
            extracted.append(translation[match.start(): match.end()])
    return "\n".join(extracted)

def flatten_dict(d):
    """
    Flattens a nested dictionary by concatenating keys with a dot.

    Parameters
    ----------
    d : dict
        The dictionary to flatten.

    Returns
    -------
    dict
        The flattened dictionary.
    """
    if isinstance(d, dict):
        return " | ".join(f"{k}: {v}" for k, v in d.items())
    return d

def normalize_key(key: str) -> str:
    """
    Normalizes a key by converting to lowercase and removing special characters.
    """
    return re.sub(r"[^a-zA-Z0-9]", "", key.lower())

# Extract JSON block from response text
def extract_json(response_text):
    """
    Extracts a JSON block from the response text.

    Parameters
    ----------
    response_text : str
        The response text from the model.

    Returns
    -------
    str
        The extracted JSON block.
    """
    match = re.search(r"```json\n(.*?)\n```", response_text, re.S)
    if match:
        return match.group(1).strip()
    return None

def parse_response(response: str, llm, query_template: str, sampling_params, pathology_report: str, retries=3) -> dict:
    """
    Parses the model's response into a standardized dictionary with simplified keys.
    Retries missing fields up to a specified number of times, including the original pathology report in follow-up queries.

    Parameters
    ----------
    response : str
        The model's response to a pathology report.
    llm : LLM
        The vLLM model instance for generating responses.
    query_template : str
        The template for generating follow-up queries.
    sampling_params : SamplingParams
        The sampling parameters for generating responses.
    pathology_report : str
        The original pathology report text.
    retries : int
        The maximum number of retries for missing fields (default is 3).

    Returns
    -------
    dict
        A dictionary containing the extracted fields with simplified keys.
    """
    # Define the fields to extract and their simplified keys
    fields = {
        "Disease Type (Short based on Unified Medical Language System)": "Disease Type",
        "Disease Type": "Disease Type",
        "Disease Type specific": "Disease Specific",
        "Is it a soft tissue tumor? (Yes/No)": "Soft Tissue Tumor",
        "Is it a soft tissue tumor?": "Soft Tissue Tumor",
        "Is it suspected or confirmed? (Suspected/Confirmed)": "Suspected Confirmed",
        "Is it suspected or confirmed?": "Suspected Confirmed",
        "Is it benign or malignant? (Benign/Malignant)": "Benign Malignant",
        "Is it benign or malignant?": "Benign Malignant",
        "What is the tumor grade? (Preferably based on the FNCLCC grading system: G1, G2, or G3. If FNCLCC grading is not available, use Low-grade or High-grade. If the tumor is benign, use 'No grade')": "Tumor Grade",
        "What is the tumor grade?": "Tumor Grade",
        "Organ in which the tumor is located (Short based on Unified Medical Language System)": "Tumor Location",
        "Organ in which the tumor is located": "Tumor Location",
        "Organ in which the tumor is located specific:": "Tumor Location Specific",
        "Approximate number of mitotic figures": "Mitotic Figures",
        "Presence of necrosis (Yes/No) and approximate extent (structured as {'Present': 'Yes/No', 'Extent': 'Description'})": "Necrosis",
        "Presence of necrosis and approximate extent": "Necrosis",
        "Presence of necrosis": "Necrosis",
        "Can you list all known mutation status? (Output structured as a list of dictionaries with 'Gene' and 'Mutation' keys)": "Mutation Status",
        "Can you list all known mutation status?": "Mutation Status",
        "Can you list all known immunohistochemistry markers? (Output structured as a list of dictionaries with 'Marker' and 'Expression' keys)": "IHC Markers",
        "Can you list all known immunohistochemistry markers?": "IHC Markers",
    }

    example = """
    Example response:
    </think>

    ```json
    {
        "Disease Type": "Myxofibrosarcoma",
        "Disease Type specific": "High-grade myxofibrosarcoma; myxo-inflammatory fibroblastic sarcoma cannot be excluded.",
        "Is it a soft tissue tumor?": "Yes",
        "Is it suspected or confirmed?": "Confirmed",
        "Is it benign or malignant?": "Malignant",
        "Tumor Grade": "High-grade",
        "Organ in which the tumor is located (Short)": "Upper limb",
        "Organ in which the tumor is located (Full)": "Solidus lateralis, right side",
        "Approximate number of mitotic figures": "2 per 10 HPF",
        "Presence of necrosis": {
            "Present": "Yes",
            "Extent": "Extensive (60%)"
        },
        "Known mutation status": [
            {"Gene": "TP53", "Mutation": "p.R248Q"},
            {"Gene": "BRAF", "Mutation": "Wild-type"}
        ]
        "Known immunohistochemistry markers": [
            {"Marker": "Desmin", "Expression": "Negative"},
            {"Marker": "HMB45", "Expression": "Negative"}
        ]
    }```
    """

    # Initialize the result dictionary
    result = {simplified_key: "Undefined" for simplified_key in fields.values()}

    # Parse the initial response
    def parse_single_response(response_text: str, result: dict, fields: dict) -> dict:
        json_block = extract_json(response_text)
        if not json_block:
            print("No JSON block found")
            return result

        try:
            json_data = json.loads(json_block)
            json_data = {k: flatten_dict(v) for k, v in json_data.items()}
            
            if len(json_data) == len(result):
                print("Direct mapping based on order")
                for simplified_key, (json_key, json_value) in zip(result.keys(), json_data.items()):
                    result[simplified_key] = json_value
            else:
                print("Fuzzy matching due to mismatch")
                normalized_fields = {normalize_key(k): v for k, v in fields.items()}
                
                for json_key, json_value in json_data.items():
                    normalized_key = normalize_key(json_key)
                    best_match = None
                    best_score = 0
                    
                    for field in normalized_fields:
                        score = fuzz.ratio(normalized_key, field)
                        if score > best_score:
                            best_score = score
                            best_match = field
                    
                    if best_match and best_score > 80:
                        result[normalized_fields[best_match]] = json_value
                    else:
                        print(f"No match found for: {json_key}")
        except json.JSONDecodeError:
            print("Response is not in JSON format")

        return result

    # Parse the initial response
    result = parse_single_response(response, result, fields)

    # Retry missing fields
    for attempt in range(retries):
        # Check which fields are still missing
        missing_fields = [k for k, v in result.items() if v == "Undefined"]
        if not missing_fields:
            break  # All fields are filled, no need to retry

        print(f"Retry {attempt + 1}/{retries}")

        # Generate a follow-up query for the missing fields
        follow_up_question = "\n".join(
            f"{i+1}. {list(fields.keys())[list(fields.values()).index(field)]}:"
            for i, field in enumerate(missing_fields)
        )

        follow_up_query = f"""
        [file name]: follow_up_{attempt}
        [file content begin]
        {pathology_report}
        [file content end]
        Extract the following information from the pathology report in a structured JSON format. If a field is not mentioned in the report, use "Not specified". Always include all fields in the output:
        {follow_up_question}
        """

        # Generate a follow-up response
        follow_up_response = llm.generate([follow_up_query], sampling_params)[0].outputs[0].text

        # Parse the follow-up response
        result = parse_single_response(follow_up_response, result, fields)

    return result

def run(data:str, patterns:str, model:str = "deepseek-ai/DeepSeek-R1", gpus=1):
    """
    Extracts structured information from pathology reports using LLM.

    Parameters
    ----------
    data : str
        Path to the CSV file containing the translated pathology reports.
    patterns : str
        Path to the JSON file containing the regular expression patterns.
    model : str
        DeepSeek model to use: V3, R1, or R1-Distill.
    gpus : int
        Number of GPUs to use for tensor parallelism.
    """
    out = Path(data)
    data = pd.read_csv(out)

    # Removing cytology:
    data = data[data["code_display"] != "Cytology report"].reset_index(drop=True)

    data = data.iloc[0:10]

    if "translation" not in data.columns:
        raise ValueError(
            "It seems like the document has not been translated... stopping here, please run translate"
        )

    # Load all regex patterns
    with open(patterns) as f:
        patterns = json.load(f)

    regex = compile_patterns(patterns)

    # Load DeepSeek model using vLLM and define sampling parameters
    llm = LLM(model=model, max_model_len=60848, trust_remote_code=True, tensor_parallel_size=gpus)
    sampling_params = SamplingParams(max_tokens=500, temperature=0.6, top_p=0.9, repetition_penalty=1.2)

    # Define question
    question = """
    Extract the following information from the pathology report in a structured JSON format. 
    If a field is not mentioned in the report, use "Not specified" exactly. Always include all fields in the output in the exact order:

    1. Disease Type (Short based on Unified Medical Language System):
    2. Disease Type specific:
    3. Is it a soft tissue tumor? (Yes/No):
    4. Is it suspected or confirmed? (Suspected/Confirmed):
    5. Is it benign or malignant? (Benign/Malignant):
    6. What is the tumor grade? (Preferably based on the FNCLCC grading system: G1, G2, or G3. If FNCLCC grading is not available, use Low-grade or High-grade. If the tumor is benign, use "No grade"):
    7. Organ in which the tumor is located (Short based on Unified Medical Language System):
    8. Organ in which the tumor is located specific:
    9. Approximate number of mitotic figures:
    10. Presence of necrosis (Yes/No) and approximate extent (structured as {"Present": "Yes/No", "Extent": "Description"}):
    11. Can you list all known mutation status? (Output structured as a list of dictionaries with "Gene" and "Mutation" keys):
    12. Can you list all known immunohistochemistry markers? (Output structured as a list of dictionaries with "Marker" and "Expression" keys):
    """

    example = """
    Example response:
    </think>

    ```json
    {
        "Disease Type": "Myxofibrosarcoma",
        "Disease Type specific": "High-grade myxofibrosarcoma; myxo-inflammatory fibroblastic sarcoma cannot be excluded.",
        "Is it a soft tissue tumor?": "Yes",
        "Is it suspected or confirmed?": "Confirmed",
        "Is it benign or malignant?": "Malignant",
        "Tumor Grade": "High-grade",
        "Organ in which the tumor is located (Short)": "Upper limb",
        "Organ in which the tumor is located (Full)": "Solidus lateralis, right side",
        "Approximate number of mitotic figures": "2 per 10 HPF",
        "Presence of necrosis": {
            "Present": "Yes",
            "Extent": "Extensive (60%)"
        },
        "Known mutation status": [
            {"Gene": "TP53", "Mutation": "p.R248Q"},
            {"Gene": "BRAF", "Mutation": "Wild-type"}
        ]
        "Known immunohistochemistry markers": [
            {"Marker": "Desmin", "Expression": "Negative"},
            {"Marker": "HMB45", "Expression": "Negative"}
        ]
    }```
    """

    queries = []
    data = data.reset_index(drop=True)
    for idx, row in data.iterrows():
        translation = row["translation"]
        if not isinstance(translation, str):
            queries.append([])
            print(f"Skipping index {idx} as the translation is not a string")
            continue

        pathology_report = extract_text(translation, regex)

        # Define query prompt
        file_name = f"report_{idx}"  # Customize file name as needed
        
        query = f"""
        [file name]: {file_name}
        [file content begin]
        {pathology_report}
        [file content end]
        {question}

        {example}
        """

        queries.append(query)

    print("Generating responses...")
    responses = llm.generate(queries, sampling_params)

    # Parse responses and add them as new columns to the DataFrame
    results = []
    for idx, response in enumerate(responses):
        response_text = response.outputs[0].text

        pathology_report = queries[idx].split("[file content begin]")[1].split("[file content end]")[0].strip()

        # Parse the response with retries for missing fields
        parsed_result = parse_response(response_text, llm, query, sampling_params, pathology_report, retries=3)
        results.append({**parsed_result, "Unparsed results": response_text})
        
    # Save results
    data = pd.concat([data, pd.DataFrame(results)], axis=1)
    output_path = out.parent / f"{out.stem}_analyzed.csv"
    data.to_csv(output_path, index=False)
    print(f"Analysis complete. Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="LLM using DeepSeek-V3 to extract information from 'verslagen' documents",
    )
    parser.add_argument(
        "-i",
        "--input",  
        required=True,
        type=str,
        help="Path to the translated documents",
    )
    parser.add_argument(
        "-r",
        "--regex",
        required=True,
        type=str,
        help="Path to the regex patterns to extract (should be .json file)",
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        type=str,
        choices=["V3", "R1", "R1-Distill"],
        help="DeepSeek model to use: V3, R1, or R1-Distill",
    )
    parser.add_argument(
        "-g",
        "--gpus",
        required=False,
        type=int,
        default=2,
        help="Number of GPUs to use for tensor parallelism",
    )
    
    args = parser.parse_args()
    model = f"deepseek-ai/DeepSeek-{args.model}"
    if args.model == "R1-Distill":
        model += "-Qwen-32B"

    run(
        data=args.input, 
        patterns=args.regex, 
        model=model, 
        gpus=args.gpus
    )


if __name__ == "__main__":
    main()
