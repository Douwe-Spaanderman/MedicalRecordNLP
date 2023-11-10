import argparse
from pathlib import Path
import os
import json
import pandas as pd
import re
import warnings

import spacy
import spacy.lang.en
import scispacy
from scispacy.linking import EntityLinker

from typing import Optional, Type

class standardized_report:
    def __init__(self, name, report:str, regex:list, bionlp13cg:Type[spacy.lang.en.English], core_sci:Optional[Type[spacy.lang.en.English]]=None, verbose:bool = False):
        self.name = name
        self.report = report
        self.regex = regex
        self.verbose = verbose
        self.bionlp13cg = bionlp13cg
        self.core_sci = core_sci
        
    def extract_patterns(self, pattern):
        regex = self.regex[pattern]
        text = regex.search(self.report)
        if not text:
            if self.verbose:
                warnings.warn(f"Pattern: {pattern} not found for {self.name}")
            return False
            
        text = text.group(0).strip()
        
        return text
    
    def run_bionlp13cg(self, text):
        return self.bionlp13cg(text)
    
    def run_core_sci(self, text):
        if self.core_sci:
            return self.core_sci(text)
        else:
            if self.verbose:
                warnings.warn(f"You tried to run the core_sci nlp model, however you did not provide it initially")
            return False
        
    def canonical_name(self, text):
        if self.core_sci:
            text = self.core_sci(text)

            #return self.core_sci(text)
        else:
            if self.verbose:
                warnings.warn(f"You tried to match canonical name using the core_sci nlp model, however you did not provide it initially")
            return text
        
    def run_clean(self, ents):
        for ent in ents:
            ent = self.canonical_name(ent)
    
    def extract_conclusion(self, text):
        if type(text) == str:
            text = self.run_bionlp13cg(text)

        phenotype = self.run_clean([ent for ent in text.ents if ent.label_ == "CANCER" and not ent.text.upper() == "TUMOR"])
        location = self.run_clean([ent for ent in text.ents if ent.label_ in ["ORGAN", "TISSUE", "ANATOMICAL_SYSTEM"]])

def compile_patterns(patterns:dict):
    regex = {}
    for name, pattern in patterns.items():
        flags = getattr(re, pattern["flags"])
        pattern = pattern["pattern"]
        regex[name] = re.compile(f"{pattern}", flags)

    return regex

def run(data, patterns, pipe):
    out = Path(data)
    data = pd.read_csv(out)

    if 'Translation' not in data.columns:
        raise ValueError("It seems like to document already has not been translated... stopping here, please run translate")

    ## Load all available NLP models
    # Specific NER
    bionlp13cg = spacy.load("en_ner_bionlp13cg_md")

    if pipe:
        with open(pipe) as f:
            pipe = json.load(f)
    
    # Core model
    core_sci = spacy.load("en_core_sci_lg")
    #core_sci.add_pipe(EntityLinker(resolve_abbreviations=True, name="hbo"))

    # Load all regex patterns
    with open(patterns) as f:
        patterns = json.load(f)

    regex = compile_patterns(patterns)

    for row, item in data[:5].iterrows():
        item = standardized_report(
            name=item["member_entity_Patient_value"], 
            report=item["Translation"], 
            regex=regex,
            bionlp13cg=bionlp13cg,
        )
        con = item.extract_patterns("Conclusion")
        doc = item.run_bionlp13cg(con)

        import ipdb
        ipdb.set_trace()


def main():
    parser = argparse.ArgumentParser(
        description="NLP on english pathology reports"
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="Path to the translated pathology data",
    )
    parser.add_argument(
        "-r",
        "--regex",
        required=True,
        type=str,
        help="Path to the regex patterns to extract",
    )
    parser.add_argument(
        "-p",
        "--pipe",
        default=None,
        type=str,
        help="Path to the pipe for additional rule based extraction",
    )

    args = parser.parse_args()
    
    run(args.input, args.regex, args.pipe)

if __name__ == "__main__":
    main()