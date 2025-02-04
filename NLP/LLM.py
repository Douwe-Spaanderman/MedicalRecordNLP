import argparse
from pathlib import Path
import os
import json
import pandas as pd
import re
import warnings
from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

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
        pattern = pattern["pattern"]e
        regex[name] = {"pattern": re.compile(f"{pattern}", flags), "start": start}

    return regex

def run(data, patterns, small=False):
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

    if small:
        tokenizer = AutoTokenizer.from_pretrained("epfl-llm/meditron-7b")
        model = AutoModelForCausalLM.from_pretrained("epfl-llm/meditron-7b")
    else:
        tokenizer = AutoTokenizer.from_pretrained("epfl-llm/meditron-70b")
        model = AutoModelForCausalLM.from_pretrained("epfl-llm/meditron-70b")

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
        Extract the following information from the pathology report:

        1. Disease Type:
        2. Is it a soft tissue tumor? (Yes/No):
        3. Is it benign or malignant?:
        4. Tumor Grade (if applicable):
        5. Organ in which the tumor is located:
        6. Approximate number of mitotic figures:
        7. Presence of necrosis (Yes/No) and approximate extent:

        Pathology Report:
        {pathology_report}
        """
        print(query)

        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        response = generator(query, max_length=256, temperature=0.2, do_sample=False)

        # Print the extracted information
        extracted_info = response[0]["generated_text"]
        print(extracted_info)

        import ipdb; ipdb.set_trace()


def main():
    parser = argparse.ArgumentParser(
        description="LLM using Meditron to extract information from 'verslagen' documents",
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
        "--small",
        action="store_true",
        default=False,
        help="Use the smaller model (7b) instead of the large one (70b)",
    )

    args = parser.parse_args()

    run(args.input, args.regex, args.small)


if __name__ == "__main__":
    main()
