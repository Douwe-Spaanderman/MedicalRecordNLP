import argparse
from pathlib import Path
import os
import json
import pandas as pd
import re
import warnings
from typing import Optional

import spacy
import spacy.matcher.matcher
import spacy.lang.en
from spacy.matcher import Matcher
from scispacy.linking import EntityLinker

from report import StandardizedReport

class PathologyReport(StandardizedReport):
    def __init__(self, name, report:str, regex:list, bionlp13cg:spacy.lang.en.English, core_sci:Optional[spacy.lang.en.English]=None, matcher:Optional[spacy.matcher.matcher.Matcher]=None, linker:Optional[EntityLinker]=None, api_key:str=None, verbose:bool = False):
        super().__init__(
            name=name, 
            report=report, 
            regex=regex, 
            bionlp13cg=bionlp13cg, 
            core_sci=core_sci, 
            matcher=matcher, 
            linker=linker, 
            api_key=api_key, 
            verbose=verbose
        )

        self.verbose = verbose
        self.phenotype = ()
        self.location = ()
        self.cell = ()
        self.grade = ()
        self.genes = ()
        self.mitosis = ()
        self.necrosis = ()

    def run_complete_pipeline(self):
        if type(self.report) == str:
            self.report = self.run_bionlp13cg(self.report)

        if not self.report or not self.report.ents:
            return False

        # Extract spans from different parts of the report
        self.extract_conclusion()
        self.extract_microscopy()
        self.extract_immunohistochemistry()
        self.extract_moleculaire()

        # For debugging see what is in compliance
        self.compliance = self.extract_patterns("Compliance")
        if self.compliance:
            import ipdb; ipdb.set_trace()

        # clean and add ents to self.report
        combined_ents = self.phenotype + self.location + self.cell + self.grade + self.genes + self.mitosis + self.necrosis
        self.report = self.clear_ents(self.report)
        self.report = self.add_ents(self.report, combined_ents)

        # Extract relations - automatically mapped to prodigy
        relations = self.add_speech_dependency(self.report, ["GRADE"], ["ADJ", "NUM", "DET"])
        relations += self.add_speech_dependency(self.report, ["NECROSIS", "MITOSIS"], ["ADJ", "NUM", "DET"], "HPF")
        relations += self.add_speech_dependency(self.report, ["GENE_OR_GENE_PRODUCT"], ["ADJ", "DET"], search_tree=True)
        
        # Map spans to prodigy
        spans = self.map_to_prodigy_spans(self.report)

        return self.map_to_prodigy(self.report, spans, relations)
        
    def extract_conclusion(self):
        conclusion = self.extract_patterns("Conclusion")
        if not conclusion:
            if self.verbose:
                warnings.warn(f"You tried to extract conclusion, however no regex match was found")
            return False

        # Phenotype has CANCER label
        self.phenotype = [ent for ent in self.report.ents if ent.label_ == "CANCER" and not ent.text.upper() == "TUMOR"]
        if not self.phenotype:
            self.phenotype = [ent for ent in self.report.ents if ent.label_ == "CELL"]

        self.phenotype = self.clean_ents_text_range(self.phenotype, conclusion)
        self.phenotype = self.clean_ents(self.phenotype)

        # Location has ORGAN/TISSUE/ANATOMICAL_SYSTEM label
        self.location = [ent for ent in self.report.ents if ent.label_ in ["ORGAN", "TISSUE", "ANATOMICAL_SYSTEM"]]
        self.location = self.clean_ents_text_range(self.location, conclusion)
        self.location = self.clean_ents(self.location)

        # Cell type based on CELL label
        if not self.cell:
            self.cell = tuple([ent for ent in self.report.ents if ent.label_ == "CELL" and not ent.text.upper() == "CELL"])
            self.cell = self.clean_ents_text_range(self.cell, conclusion)

        # Extract grade with matcher
        self.grade = self.extract_matcher_ents(self.report, label="GRADE")
        self.grade = self.clean_ents_text_range(self.grade, conclusion)

    def extract_microscopy(self):
        microscopy = self.extract_patterns("Microscopy")
        if not microscopy:
            if self.verbose:
                warnings.warn(f"You tried to extract microscopy, however no regex match was found")
            return False

        # For now clearing all not gene ents
        genes = tuple([ent for ent in self.report.ents if ent.label_ == "GENE_OR_GENE_PRODUCT"])
        genes = self.clean_ents_text_range(genes, microscopy) 
        self.genes = self.genes + genes

        # Extract grade with matcher
        if not self.grade:
            self.grade = self.extract_matcher_ents(self.report, label="GRADE")
            self.grade = self.clean_ents_text_range(self.grade, microscopy)

        # Extract mitosis with matcher
        self.mitosis = self.extract_matcher_ents(self.report, label="MITOSIS")
        self.mitosis = self.clean_ents_text_range(self.mitosis, microscopy)

        # Extract necrosis with matcher
        self.necrosis = self.extract_matcher_ents(self.report, label="NECROSIS")
        self.necrosis = self.clean_ents_text_range(self.necrosis, microscopy)


    def extract_immunohistochemistry(self):
        immunohistochemistry = self.extract_patterns("Immunohistochemistry")
        if not immunohistochemistry:
            if self.verbose:
                warnings.warn(f"You tried to extract immunohistochemistry, however no regex match was found")
            return False

        # For now clearing all not gene ents
        genes = tuple([ent for ent in self.report.ents if ent.label_ == "GENE_OR_GENE_PRODUCT"])
        genes = self.clean_ents_text_range(genes, immunohistochemistry) 
        self.genes = self.genes + genes

    def extract_moleculaire(self):
        moleculaire = self.extract_patterns("Moleculaire")
        if not moleculaire:
            if self.verbose:
                warnings.warn(f"You tried to extract moleculaire, however no regex match was found")
            return False

        # For now clearing all not gene ents
        genes = tuple([ent for ent in self.report.ents if ent.label_ == "GENE_OR_GENE_PRODUCT"])
        genes = self.clean_ents_text_range(genes, moleculaire) 
        self.genes = self.genes + genes

def compile_patterns(patterns:dict):
    regex = {}
    for name, pattern in patterns.items():
        flags = getattr(re, pattern["flags"])
        start = pattern["start"]
        pattern = pattern["pattern"]
        regex[name] = {
            "pattern" : re.compile(f"{pattern}", flags),
            "start": start
        }

    return regex

def run(data, output, patterns, pipe, api_key):
    out = Path(data)
    data = pd.read_csv(out)

    # Removing cytology:
    data = data[data["code_display"] != "Cytology report"]

    if 'Translation' not in data.columns:
        raise ValueError("It seems like to document already has not been translated... stopping here, please run translate")

    ## Load all available NLP models
    # Specific NER
    bionlp13cg = spacy.load("en_ner_bionlp13cg_md")

    # EntityLinker
    linker = EntityLinker(resolve_abbreviations=True, name="umls")
    bionlp13cg.add_pipe(linker)

    # Matcher
    matcher = Matcher(bionlp13cg.vocab)
    if pipe:
        with open(pipe, "r") as f:
            pipe = json.load(f)
            pipe = {key: value for key, value in pipe}

        for rule_name, rule in pipe.items():
            matcher.add(rule_name, rule)

    # Load all regex patterns
    with open(patterns) as f:
        patterns = json.load(f)

    regex = compile_patterns(patterns)

    analyzed_reports = []
    labels = []
    data = data.reset_index(drop=True)
    for row, item in data[0:2].iterrows():
        report = PathologyReport(
            name=item["member_entity_Patient_value"], 
            report=item["Translation"], 
            regex=regex,
            bionlp13cg=bionlp13cg,
            #core_sci=core_sci,
            matcher=matcher,
            linker=linker,
            api_key=api_key
        )

        analyzed_report = report.run_complete_pipeline()

        labels.extend([x["label"] for x in analyzed_report["spans"]])
        analyzed_reports.append(analyzed_report)

    # Save as json lines format
    output = Path(output)
    with open(output.with_suffix('.jsonl'), 'w') as f:
        for item in analyzed_reports:
            json.dump(item, f)
            f.write('\n')

    # Save unique labels
    labels = list(set(labels))
    with open(output.with_suffix('.txt'), 'w') as f:
        for label in labels:
            f.write(f"{label}\n")

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
        "-o",
        "--output",
        required=True,
        type=str,
        help="Path to the save analyzed reports (as jsonl)",
    )
    parser.add_argument(
        "-r",
        "--regex",
        required=True,
        type=str,
        help="Path to the regex patterns to extract (should be .json file)",
    )
    parser.add_argument(
        "-p",
        "--pipe",
        default=None,
        type=str,
        help="Path to the pipe for additional rule based extraction (should be .py file)",
    )
    parser.add_argument(
        "-k",
        "--api",
        default=False,
        type=str,
        help="API key to access UMLS, if False will look into environment variable named 'UMLS'",
    )

    args = parser.parse_args()

    api_key = args.api
    if not api_key:
        try:
            api_key = os.environ["UMLS"]
        except KeyError:
            warnings.warn("API key not provided and not found in environment variable, UMLS api will be disabled")
            api_key = None
        
    run(args.input, args.output, args.regex, args.pipe, api_key)

if __name__ == "__main__":
    main()