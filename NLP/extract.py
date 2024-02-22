import argparse
from pathlib import Path
import os
import json
import pandas as pd
import re
import warnings
import copy
import requests
import numpy as np

import spacy
import spacy.tokens.span
import spacy.tokens.doc
import spacy.matcher.matcher
import spacy.lang.en
from spacy.tokens import Span
Span.set_extension("linker", default=None)
from spacy.matcher import Matcher
import scispacy
from scispacy.linking import EntityLinker

from thefuzz import fuzz

from typing import Optional, Union, List, Dict, Tuple

class StandardizedReport:
    def __init__(self, name, report:str, regex:list, bionlp13cg:spacy.lang.en.English, core_sci:Optional[spacy.lang.en.English]=None, matcher:Optional[spacy.matcher.matcher.Matcher]=None, linker:Optional[EntityLinker]=None, api_key:str=None, verbose:bool = False):
        self.name = name
        self.report = report
        self.regex = regex
        self.verbose = verbose
        self.bionlp13cg = bionlp13cg
        self.core_sci = core_sci
        self.matcher = matcher
        self.linker = linker
        self.api_key = api_key

        # replace all the spaces
        self.report = self.remove_spaces(self.report)

    def remove_spaces(self, text:str) -> str:
        text = re.sub(r'\r\n', r'\n', text)
        text = re.sub(r'(\n)', r' ', text)
        text = re.sub(r'( ){2,}', r'\n', text)
        return text

    def run_bionlp13cg(self, text:str) -> spacy.tokens.doc.Doc:
        """
        Text to Doc using bionlp13cg scispacy model
        """
        return self.bionlp13cg(text)
    
    def run_core_sci(self, text:str) -> Union[bool, spacy.tokens.doc.Doc]:
        """
        Text to Doc using core_sci scispacy model
        """
        if self.core_sci:
            return self.core_sci(text)
        else:
            if self.verbose:
                warnings.warn(f"You tried to run the core_sci nlp model, however you did not provide it initially")
            return False

    def run_matcher(self, doc:spacy.tokens.doc.Doc) -> Union[bool, spacy.tokens.doc.Doc]:
        """
        Run matcher on Doc object
        """
        if self.matcher:
            return self.matcher(doc)
        else:
            if self.verbose:
                warnings.warn(f"You tried to run the rule based matcher, however you did not provide it initially")
            return False
        
    def extract_patterns(self, pattern:str) -> Tuple[int]:
        """
        Extract specific text from the whole report based on regex pattern
        UPDATED: now only returns start - finish idx
        """
        regex = self.regex[pattern]["pattern"]
        output = regex.search(self.report.text)
        if not output:
            if self.verbose:
                warnings.warn(f"Pattern: {pattern} not found for {self.name}")
            return False
            
        start, end = output.span()
        
        return start, end
    
    def extract_matcher_ents(self, doc:spacy.tokens.doc.Doc, label:Union[str, List[str]]="all"):
        """
        Run the matcher and add the ents to doc. label can be used to specifiy which ent you want to find.
        UPDATED: now returns ents instead of doc
        """
        matches = self.run_matcher(doc)

        ents = ()
        for match_id, start, end in matches:
            entity_label = self.bionlp13cg.vocab.strings[match_id]
            if label != "all":
                if type(label) == str:
                    label = [label]
                
                if entity_label not in label:
                    continue

            ent = Span(doc, start, end, label=entity_label)
            ents = ents + (ent,)
        
        return ents

    def clear_ents(self, doc:spacy.tokens.doc.Doc) -> spacy.tokens.doc.Doc:
        """
        This clears/deletes all the ents found in a Doc object, useful after cleaning ents
        """
        doc.ents = ()
        return doc

    def add_ents(self, doc:spacy.tokens.doc.Doc, ents:tuple) -> spacy.tokens.doc.Doc:
        """
        Adding Span ent objects to Doc. Best to do this to clear doc!

        ents is tuple[pacy.tokens.span.Span], somehow wasn't able to do typing this way?
        """
        for ent in ents:
            doc = self.add_ent(ent, doc)
            
        return doc

    def add_ent(self, ent:spacy.tokens.span.Span, doc:spacy.tokens.doc.Doc) -> spacy.tokens.doc.Doc:
        """
        Adding Span ent objects to Doc, also overwritting overlapping ents in initial Doc.
        """
        if any(ent.start <= ent_.start < ent.end or ent.start < ent_.end <= ent.end for ent_ in doc.ents):
            doc.ents = (ent_ for ent_ in doc.ents if not (ent.start <= ent_.start < ent.end or ent.start < ent_.end <= ent.end))

        doc.ents = doc.ents + (ent,)
        return doc

    def add_speech_dependency(self, doc:spacy.tokens.doc.Doc, label:Union[str, List[str]], dependencies:Union[str, List[str]], additional_token:Optional[Union[str, List[str]]] = False, search_tree:bool = False):
        """
        Adding part-of-speech dependencies for specific labels in Doc, e.g. label cancer or mitose, 
        based on dependency tags, e.g. ADJ (adjective), NUM (numeral).
        Note you can also provide an additional token to search for extra connections, this is specificially done for example with Mitose and necrose with "HPF"
        """
        if type(label) == str:
            label = [label]

        if type(dependencies) == str:
            dependencies = [dependencies]

        if type(additional_token) == str:
            additional_token = [additional_token]

        relations = []
        ents = [ent for ent in doc.ents if ent.label_ in label]
        for ent in ents:
            relation_found = False
            for token in ent.root.children:
                relations, relation_found = self.check_dependency(ent, token, dependencies, additional_token, relations, relation_found)

            if not relation_found:
                for token in ent.root.ancestors:
                    relations, relation_found = self.check_dependency(ent, token, dependencies, additional_token, relations, relation_found)

            if search_tree and not relation_found: 
                for token in ent.root.ancestors:
                    for t in token.children:
                        relations, relation_found = self.check_dependency(ent, t, dependencies, additional_token, relations, relation_found)
            
        return relations

    def check_dependency(self, ent, token, dependencies:Union[str, List[str]], additional_token:Optional[Union[str, List[str]]] = False, relations:list = [], relation_found:bool = False):
        """
        checks token dependencies against ent by seeing if dependency matches, or in case addititional token with manual check.
        """
        if additional_token:
            # Additional token, such as HPF to extract connections
            for additional in additional_token:
                if additional in token.text.upper() or (hasattr(token, 'label_') and additional in token.label_):
                    relations.append(self.map_to_prodigy_relation(token, ent))
                    # Also finds the linked words for the additional token as it also relates to ent
                    for child in token.children:
                        if child.pos_ in dependencies:
                            relations.append(self.map_to_prodigy_relation(child, ent))

                    relation_found = True
        
        if token.pos_ in dependencies:
            relations.append(self.map_to_prodigy_relation(token, ent))
            relation_found = True

        return relations, relation_found

    def run_api(self, endpoint:str, entry:str="https://uts-ws.nlm.nih.gov/rest/content", version:str="current"):
        """
        Run UMLS API
        """
        if not self.api_key:
            if self.verbose:
                warnings.warn(f"You tried to run the the UMLS API, however you did not provide an api key")
            return False

        # Some sanitization of input
        if endpoint.startswith("/"):
            endpoint = endpoint[1:]

        if endpoint.endswith("/"):
            endpoint = endpoint[:-1]

        url = f'{entry}/{version}/{endpoint}?pageSize=1000&apiKey={self.api_key}'
        response = requests.get(url)
        return response.json()

    def collect_ancestors(self, cui:str, database:str="SNOMED_CT"):
        """
        Extract ancestor diseases based on CUI for UMLS API mapped to hierarchical database, e.g. MSH or SNOMED_CT
        """
        atoms = self.run_api(f"CUI/{cui}/atoms")
        CUI_mapped = [x for x in atoms["result"] if x["rootSource"] == database]

        mapped = []
        for item in CUI_mapped:
            id_ = item["sourceConcept"].split("/")[-1]
            mapped.extend(self.run_api(f"source/{database}/{id_}/ancestors")["result"])

        return mapped
        
    def find_canonical_name(self, ent:spacy.tokens.span.Span):
        """
        Find the best matching canonical name based on a linker. Note, only tested for UMLS as it is most comprehensive.
        """
        if self.linker:
            if not ent._.kb_ents:
                return None

            # Take only highest scoring linker match, sometimes it has multiple with same score (often pscore = 1)
            idx = 0
            pscore = 0
            for link in ent._.kb_ents:
                if pscore <= link[1]:
                    idx += 1
                    pscore = link[1]

            hidx = 0
            hscore = 0
            if idx != 1:
                # For dealing with multiple linker matches, I made this fuzz match to find the "better" match out of the options
                # Note, sort ratio has the best effect, as set ratio doesn't filter to detailed matches, e.g. pediatric sarcoma for sarcoma
                for i, link in enumerate(ent._.kb_ents[:idx]):
                    linked_ent = self.linker.kb.cui_to_entity[link[0]]
                    cscore = fuzz.token_sort_ratio(ent.text, linked_ent.canonical_name)
                    if cscore > hscore:
                        hscore = cscore
                        hidx = i

            linked_ent = self.linker.kb.cui_to_entity[ent._.kb_ents[hidx][0]]
            return {"uid": linked_ent.concept_id, "name": linked_ent.canonical_name}
        else:
            if self.verbose:
                warnings.warn(f"You tried to match canonical name using entitylinker, however you did not provide it initially")
            return text
        
    def clean_ents_text_range(self, ents:List[spacy.tokens.span.Span], text_range:Tuple[int]):
        """
        Cleans ents based on a specific text range, i.e. only in parts of the documents
        """
        new_ents = ()
        for idx, ent in enumerate(ents):
            if ent.start_char >= text_range[0] and ent.start_char <= text_range[1]:
                new_ents = new_ents + (ent,)

        return new_ents

    def clean_ents(self, ents:List[spacy.tokens.span.Span]):
        """
        Cleans from similar ents, e.g. Liposarcoma and Liposarcoma Knee. Cleans using the linker and fuzz
        """
        new_ents = ()
        for idx, ent in enumerate(ents):
            link = self.find_canonical_name(ent)
            # Removing objects not found in linker
            if not link:
                continue
            else:
                ent._.linker = link
                new_ents = new_ents + (ent,)

        return new_ents

    def filter_disease(self, cleaned_ents:Dict[str, str], parent_disease:Union[str, List[str]]):
        """
        filters the disease/phenotypes based on UMLS parent
        """
        for ent in cleaned_ents:
            # Special cases: C0238198 (GIST) C3179349 (GIST)

            ancestors = self.collect_ancestors(ent["conceptID"])
            ## WORK IN PROGRESS ##

        return cleaned_ents

    def map_to_prodigy_relation(self, head:spacy.tokens.token.Token, child:spacy.tokens.span.Span):
        '''
        Map relations ship, i.e. head and child to prodigy format
        '''
        return {
            "head": head.i,
            "child": child.start,
            "label": child.label_,
            "head_span": {"start": head.idx, "end": head.idx+len(head), "token_start": head.i, "token_end": head.i, "label": None},
            "child_span": {"start": child.start_char, "end": child.end_char, "token_start": child.start, "token_end": child.end-1, "label": child.label_}
        }

    def map_to_prodigy_span(self, ent:spacy.tokens.span.Span):
        '''
        Map span to prodigy format
        '''
        return {
            "start": ent.start_char, 
            "end": ent.end_char, 
            "token_start": ent.start,
            "token_end": ent.end-1,
            "label": ent.label_,
            "linker": ent._.linker
        }

    def map_to_prodigy_spans(self, doc:spacy.tokens.doc.Doc):
        '''
        Doc wrapper for mapping spans to prodigy format
        '''
        spans = []
        for ent in doc.ents:
            spans.append(self.map_to_prodigy_span(ent))
        return spans

    def map_to_prodigy(self, doc:spacy.tokens.doc.Doc, ent:List[dict], relations:Optional[List[dict]] = False):
        '''
        Doc wrapper for mapping spans to prodigy format
        '''
        tokens = []
        for token in doc:
            tokens.append(
                {
                    "text" : token.text,
                    "start" : token.idx,
                    "end" : token.idx+len(token),
                    "id" : token.i,
                    "ws" : bool(token.whitespace_),
                    "disabled": token.is_punct or token.is_space or token.is_bracket
                }
            )

        if relations:
            return {
                "text": doc.text,
                "tokens": tokens,
                "spans": ent,
                "relations": relations
            }
        else:
            return {
                "text": doc.text,
                "tokens": tokens,
                "spans": ent
            }

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

    # Core model
    #core_sci = spacy.load("en_core_sci_lg")

    # EntityLinker
    linker = EntityLinker(resolve_abbreviations=True, name="umls")
    #core_sci.add_pipe(linker)
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