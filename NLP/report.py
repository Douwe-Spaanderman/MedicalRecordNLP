import json
import re
import warnings
import requests
from typing import Optional, Union, List, Dict, Tuple

import spacy
import spacy.tokens.span
import spacy.tokens.doc
import spacy.matcher.matcher
import spacy.lang.en
from spacy.tokens import Span

Span.set_extension("linker", default=None)
from spacy.matcher import Matcher
from scispacy.linking import EntityLinker

from thefuzz import fuzz


class StandardizedReport:
    def __init__(
        self,
        name,
        report: str,
        regex: list,
        bionlp13cg: spacy.lang.en.English,
        core_sci: Optional[spacy.lang.en.English] = None,
        matcher: Optional[spacy.matcher.matcher.Matcher] = None,
        linker: Optional[EntityLinker] = None,
        api_key: str = None,
        verbose: bool = False,
    ):
        """
        Initialize the StandardizedReport class.

        Parameters
        ----------
        name : str
            Name identifier for the report.
        report : str
            The report text to process.
        regex : list
            List of regular expressions for text extraction.
        bionlp13cg : spacy.lang.en.English
            SciSpacy model for NLP processing.
        core_sci : Optional[spacy.lang.en.English]
            Optional core_sci SciSpacy model.
        matcher : Optional[spacy.matcher.matcher.Matcher]
            Optional matcher for rule-based entity recognition.
        linker : Optional[EntityLinker]
            Optional entity linker for canonical name matching.
        api_key : str, optional
            API key for external services.
        verbose : bool, default False
            Verbose mode for warnings and notifications.
        """
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

    def remove_spaces(self, text: str) -> str:
        """
        Remove extra spaces and line breaks in the text.

        Parameters
        ----------
        text : str
            Input text to process.

        Returns
        -------
        str
            Cleaned text without extra spaces and line breaks.
        """
        text = re.sub(r"\r\n", r"\n", text)
        text = re.sub(r"(\n)", r" ", text)
        text = re.sub(r"( ){2,}", r"\n", text)
        return text

    def run_bionlp13cg(self, text: str) -> spacy.tokens.doc.Doc:
        """
        Process text using the bionlp13cg SciSpacy model.

        Parameters
        ----------
        text : str
            Input text for processing.

        Returns
        -------
        spacy.tokens.doc.Doc
            NLP processed Doc object.
        """
        return self.bionlp13cg(text)

    def run_core_sci(self, text: str) -> Union[bool, spacy.tokens.doc.Doc]:
        """
        Process text using the core_sci SciSpacy model.

        Parameters
        ----------
        text : str
            Input text for processing.

        Returns
        -------
        Union[bool, spacy.tokens.doc.Doc]
            Processed Doc if core_sci is provided, otherwise False.
        """
        if self.core_sci:
            return self.core_sci(text)
        else:
            if self.verbose:
                warnings.warn(
                    f"You tried to run the core_sci nlp model, however you did not provide it initially"
                )
            return False

    def run_matcher(
        self, doc: spacy.tokens.doc.Doc
    ) -> Union[bool, spacy.tokens.doc.Doc]:
        """
        Run the matcher on the provided Doc object.

        Parameters
        ----------
        doc : spacy.tokens.doc.Doc
            Doc object to apply the matcher on.

        Returns
        -------
        Union[bool, spacy.tokens.doc.Doc]
            Matcher result or False if not initialized.
        """
        if self.matcher:
            return self.matcher(doc)
        else:
            if self.verbose:
                warnings.warn(
                    f"You tried to run the rule based matcher, however you did not provide it initially"
                )
            return False

    def extract_patterns(self, pattern: str) -> Tuple[int]:
        """
        Extract text ranges using regex pattern.

        Parameters
        ----------
        pattern : str
            Regular expression pattern to extract.

        Returns
        -------
        Tuple[int]
            Start and end index of the matched pattern.
        """
        regex = self.regex[pattern]["pattern"]
        output = regex.search(self.report.text)
        if not output:
            if self.verbose:
                warnings.warn(f"Pattern: {pattern} not found for {self.name}")
            return False

        start, end = output.span()

        return start, end

    def extract_matcher_ents(
        self, doc: spacy.tokens.doc.Doc, label: Union[str, List[str]] = "all"
    ):
        """
        Extract entities using matcher with label filtering.

        Parameters
        ----------
        doc : spacy.tokens.doc.Doc
            NLP Doc to search entities within.
        label : Union[str, List[str]], optional
            Entity label(s) to extract. Default is "all".

        Returns
        -------
        tuple
            Extracted entities.
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

    def clear_ents(self, doc: spacy.tokens.doc.Doc) -> spacy.tokens.doc.Doc:
        """
        Remove all entities from a Doc object.

        Parameters
        ----------
        doc : spacy.tokens.doc.Doc
            Doc object to clear entities from.

        Returns
        -------
        spacy.tokens.doc.Doc
            Doc with no entities.
        """
        doc.ents = ()
        return doc

    def add_ents(self, doc: spacy.tokens.doc.Doc, ents: tuple) -> spacy.tokens.doc.Doc:
        """
        Add entities to a Doc object.

        Parameters
        ----------
        doc : spacy.tokens.doc.Doc
            Target Doc object.
        ents : tuple
            Entities to add.

        Returns
        -------
        spacy.tokens.doc.Doc
            Doc with added entities.
        """
        for ent in ents:
            doc = self.add_ent(ent, doc)

        return doc

    def add_ent(
        self, ent: spacy.tokens.span.Span, doc: spacy.tokens.doc.Doc
    ) -> spacy.tokens.doc.Doc:
        """
        Add a single entity to a Doc, overwriting overlaps.

        Parameters
        ----------
        ent : spacy.tokens.span.Span
            Entity span to add.
        doc : spacy.tokens.doc.Doc
            Target Doc object.

        Returns
        -------
        spacy.tokens.doc.Doc
            Doc with added entity.
        """
        if any(
            (ent.start <= ent_.start < ent.end or ent.start < ent_.end <= ent.end) or
            (ent_.start <= ent.start < ent_.end or ent_.start < ent.end <= ent_.end)
            for ent_ in doc.ents
        ):
            doc.ents = (
                ent_
                for ent_ in doc.ents
                if not (
                    (ent.start <= ent_.start < ent.end or ent.start < ent_.end <= ent.end) or
                    (ent_.start <= ent.start < ent_.end or ent_.start < ent.end <= ent_.end)
                )
            )

        doc.ents = doc.ents + (ent,)
        return doc

    def add_speech_dependency(
        self,
        doc: spacy.tokens.doc.Doc,
        label: Union[str, List[str]],
        dependencies: Union[str, List[str]],
        additional_token: Optional[Union[str, List[str]]] = False,
        search_tree: bool = False,
    ) -> List[dict]:
        """
        Adds part-of-speech dependencies for specific labels in a Doc, such as "cancer" or "mitose",
        based on dependency tags like ADJ (adjective) or NUM (numeral). An additional token (e.g., "HPF")
        can also be provided to search for further connections between entities and tokens. Optionally,
        the function can search the entire dependency tree.

        Parameters
        ----------
        doc : spacy.tokens.doc.Doc
            The processed document containing entities.
        label : Union[str, List[str]]
            The labels of the entities to target (e.g., "cancer").
        dependencies : Union[str, List[str]]
            The dependency tags to search for (e.g., ADJ).
        additional_token : Optional[Union[str, List[str]]], optional
            Additional tokens to search for extra connections (e.g., "HPF"), by default False.
        search_tree : bool, optional
            Whether to search the entire dependency tree, by default False.

        Returns
        -------
        List[dict]
            A list of relations between tokens and entities, formatted for Prodigy.
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
                relations, relation_found = self.check_dependency(
                    ent,
                    token,
                    dependencies,
                    additional_token,
                    relations,
                    relation_found,
                )

            if not relation_found:
                for token in ent.root.ancestors:
                    relations, relation_found = self.check_dependency(
                        ent,
                        token,
                        dependencies,
                        additional_token,
                        relations,
                        relation_found,
                    )

            if search_tree and not relation_found:
                for token in ent.root.ancestors:
                    for t in token.children:
                        relations, relation_found = self.check_dependency(
                            ent,
                            t,
                            dependencies,
                            additional_token,
                            relations,
                            relation_found,
                        )

        return relations

    def check_dependency(
        self,
        ent,
        token,
        dependencies: Union[str, List[str]],
        additional_token: Optional[Union[str, List[str]]] = False,
        relations: list = [],
        relation_found: bool = False,
    ) -> Tuple[List[dict], bool]:
        """
        Checks if a token's part-of-speech tag matches the provided dependencies for a given entity.
        If an additional token is provided, it also checks manually for connections. If a relation
        is found, it appends it to the relations list.

        Parameters
        ----------
        ent : spacy.tokens.span.Span
            The entity being checked.
        token : spacy.tokens.token.Token
            The token whose dependency is being checked.
        dependencies : Union[str, List[str]]
            A list of part-of-speech tags to match (e.g., ADJ).
        additional_token : Optional[Union[str, List[str]]], optional
            Additional tokens to check manually, by default False.
        relations : list, optional
            List to append found relations to, by default an empty list.
        relation_found : bool, optional
            Boolean indicating if a relation has already been found, by default False.

        Returns
        -------
        Tuple[List[dict], bool]
            Updated relations list and the relation_found status.
        """
        if additional_token:
            # Additional token, such as HPF to extract connections
            for additional in additional_token:
                if additional in token.text.upper() or (
                    hasattr(token, "label_") and additional in token.label_
                ):
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

    def run_api(
        self,
        endpoint: str,
        entry: str = "https://uts-ws.nlm.nih.gov/rest/content",
        version: str = "current",
    ) -> dict:
        """
        Runs a request to the UMLS API at the specified endpoint and version.
        Returns the API response as JSON.

        Parameters
        ----------
        endpoint : str
            The specific API endpoint to query.
        entry : str, optional
            The base URL of the API (default is "https://uts-ws.nlm.nih.gov/rest/content").
        version : str, optional
            The API version to use (default is "current").

        Returns
        -------
        dict
            The response from the API in JSON format.
        """
        if not self.api_key:
            if self.verbose:
                warnings.warn(
                    f"You tried to run the the UMLS API, however you did not provide an api key"
                )
            return False

        # Some sanitization of input
        if endpoint.startswith("/"):
            endpoint = endpoint[1:]

        if endpoint.endswith("/"):
            endpoint = endpoint[:-1]

        url = f"{entry}/{version}/{endpoint}?pageSize=1000&apiKey={self.api_key}"
        response = requests.get(url)
        return response.json()

    def collect_ancestors(self, cui: str, database: str = "SNOMED_CT") -> List[dict]:
        """
        Retrieves the ancestors of a given Concept Unique Identifier (CUI)
        from a specified hierarchical database (e.g., SNOMED_CT or MSH) using the UMLS API.

        Parameters
        ----------
        cui : str
            The CUI for which ancestors are to be retrieved.
        database : str, optional
            The hierarchical database to use (default is "SNOMED_CT").

        Returns
        -------
        List[dict]
            A list of ancestors associated with the provided CUI in the specified database.
        """
        atoms = self.run_api(f"CUI/{cui}/atoms")
        CUI_mapped = [x for x in atoms["result"] if x["rootSource"] == database]

        mapped = []
        for item in CUI_mapped:
            id_ = item["sourceConcept"].split("/")[-1]
            mapped.extend(self.run_api(f"source/{database}/{id_}/ancestors")["result"])

        return mapped

    def find_canonical_name(self, ent: spacy.tokens.span.Span) -> Optional[dict]:
        """
        Finds the best matching canonical name for an entity using a knowledge base linker.
        If multiple matches are found, fuzzy matching is used to determine the best match.

        Parameters
        ----------
        ent : spacy.tokens.span.Span
            The entity for which to find the canonical name.

        Returns
        -------
        dict or None
            A dictionary with the UID and canonical name of the entity, or None if no match is found.
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
                warnings.warn(
                    f"You tried to match canonical name using entitylinker, however you did not provide it initially"
                )
            return text

    def clean_ents_text_range(
        self, ents: List[spacy.tokens.span.Span], text_range: Tuple[int]
    ) -> Tuple[spacy.tokens.span.Span]:
        """
        Filters a list of entities to include only those within a specified range of the text.

        Parameters
        ----------
        ents : List[spacy.tokens.span.Span]
            The list of entities to filter.
        text_range : Tuple[int]
            The start and end character positions for the text range.

        Returns
        -------
        Tuple[spacy.tokens.span.Span]
            The filtered entities that fall within the specified text range.
        """
        new_ents = ()
        for idx, ent in enumerate(ents):
            if ent.start_char >= text_range[0] and ent.start_char <= text_range[1]:
                new_ents = new_ents + (ent,)

        return new_ents

    def clean_ents(
        self, ents: List[spacy.tokens.span.Span]
    ) -> Tuple[spacy.tokens.span.Span]:
        """
        Cleans a list of entities by removing duplicates and irrelevant entries.
        Uses canonical name matching via a linker and fuzzy matching to determine
        the best match for each entity.

        Parameters
        ----------
        ents : List[spacy.tokens.span.Span]
            The list of entities to clean.

        Returns
        -------
        Tuple[spacy.tokens.span.Span]
            A tuple of cleaned entities.
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

    def filter_disease(
        self, cleaned_ents: Dict[str, str], parent_disease: Union[str, List[str]]
    ) -> Dict[str, str]:
        """
        Filters a list of diseases or phenotypes based on their UMLS parent terms
        by collecting ancestor terms for the provided entities.

        Parameters
        ----------
        cleaned_ents : Dict[str, str]
            A dictionary of cleaned entities with concept IDs.
        parent_disease : Union[str, List[str]]
            The parent disease(s) used to filter the entities.

        Returns
        -------
        Dict[str, str]
            The filtered dictionary of diseases or phenotypes.
        """
        for ent in cleaned_ents:
            # Special cases: C0238198 (GIST) C3179349 (GIST)

            ancestors = self.collect_ancestors(ent["conceptID"])
            raise NotImplementedError

        return cleaned_ents

    def map_to_prodigy_relation(
        self, head: spacy.tokens.token.Token, child: spacy.tokens.span.Span
    ) -> dict:
        """
        Maps a head-child relationship between tokens into Prodigy format,
        including their span and token indices.

        Parameters
        ----------
        head : spacy.tokens.token.Token
            The head token in the relationship.
        child : spacy.tokens.span.Span
            The child span in the relationship.

        Returns
        -------
        dict
            A dictionary representing the relationship in Prodigy format.
        """
        return {
            "head": head.i,
            "child": child.start,
            "label": child.label_,
            "head_span": {
                "start": head.idx,
                "end": head.idx + len(head),
                "token_start": head.i,
                "token_end": head.i,
                "label": None,
            },
            "child_span": {
                "start": child.start_char,
                "end": child.end_char,
                "token_start": child.start,
                "token_end": child.end - 1,
                "label": child.label_,
            },
        }

    def map_to_prodigy_span(self, ent: spacy.tokens.span.Span) -> dict:
        """
        Maps an entity span into Prodigy format, including start and end
        character positions, token indices, and labels.

        Parameters
        ----------
        ent : spacy.tokens.span.Span
            The entity span to map.

        Returns
        -------
        dict
            A dictionary representing the span in Prodigy format.
        """
        return {
            "start": ent.start_char,
            "end": ent.end_char,
            "token_start": ent.start,
            "token_end": ent.end - 1,
            "label": ent.label_,
            "linker": ent._.linker,
        }

    def map_to_prodigy_spans(self, doc: spacy.tokens.doc.Doc) -> List[dict]:
        """
        Maps all entity spans in a document to Prodigy format.

        Parameters
        ----------
        doc : spacy.tokens.doc.Doc
            The document containing entity spans.

        Returns
        -------
        List[dict]
            A list of dictionaries representing the spans in Prodigy format.
        """
        spans = []
        for ent in doc.ents:
            spans.append(self.map_to_prodigy_span(ent))
        return spans

    def out_of_bounds(
        self,
        token: spacy.tokens.token.Token,
        mute: Optional[List[tuple]] = None
    ) -> bool:
        """
        Checks if the token is in mute (i.e. and not out of bounds). 
        If not in mute (i.e. out of bounds), disable token.

        Parameters
        ----------
        token : spacy.tokens.token.Token
            The token to check.
        mute : Optional[List[tuple]], optional
            A list of start-end of structured reporting format to disable out of bound tokens.

        Returns
        -------
        bool
            True or False if token in any object of mute
        """
        if mute:
            return all([not (start <= token.idx <= end) for start, end in mute])
        else:
            return False

    def map_to_prodigy(
        self,
        doc: spacy.tokens.doc.Doc,
        ent: List[dict],
        relations: Optional[List[dict]] = None,
        mute: Optional[List[tuple]] = None
    ) -> dict:
        """
        Maps the tokens, entity spans, and optionally relationships of a document
        into Prodigy format.

        Parameters
        ----------
        doc : spacy.tokens.doc.Doc
            The document to map.
        ent : List[dict]
            The entity spans in Prodigy format.
        relations : Optional[List[dict]], optional
            A list of relationships in Prodigy format, by default False.
        mute : Optional[List[tuple]], optional
            A list of start-end of structured reporting format to disable out of bound tokens.

        Returns
        -------
        dict
            A dictionary representing the document in Prodigy format.
        """
        tokens = []
        for token in doc:
            tokens.append(
                {
                    "text": token.text,
                    "start": token.idx,
                    "end": token.idx + len(token),
                    "id": token.i,
                    "ws": bool(token.whitespace_),
                    "disabled": token.is_punct or token.is_space or token.is_bracket or self.out_of_bounds(token, mute),
                }
            )

        if relations:
            return {
                "text": doc.text,
                "tokens": tokens,
                "spans": ent,
                "relations": relations,
            }
        else:
            return {"text": doc.text, "tokens": tokens, "spans": ent}
