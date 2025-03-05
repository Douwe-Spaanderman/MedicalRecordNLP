import argparse
from pathlib import Path
import os
import json
import pandas as pd
import re
import warnings
from typing import Optional

from vllm import LLM, SamplingParams

def compile_patterns(patterns: dict) -> dict:
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
    regex = {}
    for name, pattern in patterns.items():
        flags = getattr(re, pattern["flags"])
        start = pattern["start"]
        pattern = pattern["pattern"]
        regex[name] = {"pattern": re.compile(f"{pattern}", flags), "start": start}

    return regex

def load_deepseek(model_name: str = "deepseek-ai/deepseek-v3"):
    """
    Loads the DeepSeek-V3 model using vLLM.
    """
    llm = LLM(model=model_name, trust_remote_code=True, tensor_parallel_size=1)  # Adjust tensor_parallel_size based on GPU count
    return llm

def generate_response(llm, query, max_tokens=500, temperature=0.2, top_p=0.9, repetition_penalty=1.2):
    """
    Generates a response using DeepSeek-V3 with vLLM.
    """
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    responses = llm.generate([query], sampling_params)
    return responses[0].outputs[0].text

def run(data, patterns):
    out = Path(data)
    data = pd.read_csv(out)

    # Removing cytology:
    data = data[data["code_display"] != "Cytology report"]

    if "translation" not in data.columns:
        raise ValueError(
            "It seems like the document has not been translated... stopping here, please run translate"
        )

    # Load all regex patterns
    with open(patterns) as f:
        patterns = json.load(f)

    regex = compile_patterns(patterns)

    # Load DeepSeek-V3 model using vLLM
    llm = load_deepseek()

    analyzed_reports = []
    data = data.reset_index(drop=True)
    for idx, row in data.iterrows():
        translation = row["translation"]
        if not isinstance(translation, str):
            analyzed_reports.append([])
            print(f"Skipping index {idx} as the translation is not a string")
            continue

        pathology_report = ""
        for pattern in regex.values():
            matches = pattern["pattern"].search(translation)
            if matches:
                start, end = matches.span()
                pathology_report += translation[start:end]

        # Define query prompt
        query = f"""
        Extract the following information (if reported) from the pathology report:

        1. Disease Type (Short based on Unified Medical Language System):
        2. Disease Type specific:
        3. Is it a soft tissue tumor? (Yes/No):
        4. Is it suspected or confirmed? (Suspected/Confirmed):
        5. Is it benign or malignant? (Benign/Malignant):
        6. What is the tumor grade?:
        7. Organ in which the tumor is located:
        8. Approximate number of mitotic figures:
        9. Presence of necrosis (Yes/No) and approximate extent:
        10. Can you list all known mutation status? (output structured should be geneX: mutationX | geneY: mutationY):
        11. Can you list all known immunohistochemistry markers? (output structured should be markerX: expressionX | markerY: expressionY):

        Pathology Report:
        "{pathology_report}"
        """
        # Generate response using vLLM
        try:
            response = generate_response(
                llm,
                query,
                max_tokens=500,
                temperature=0.2,
                top_p=0.9,
                repetition_penalty=1.2,
            )
            print(response)
            analyzed_reports.append(response)
        except Exception as e:
            print(f"Error generating response for index {idx}: {e}")
            analyzed_reports.append("Error")

    # Save results
    data["analyzed_report"] = analyzed_reports
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
    
    args = parser.parse_args()

    run(args.input, args.regex)


if __name__ == "__main__":
    main()
