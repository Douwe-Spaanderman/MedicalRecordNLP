import spacy
from spacy.matcher import Matcher
from spacy import displacy

nlp = spacy.load("en_ner_bionlp13cg_md")

#nlp = spacy.load("en_core_sci_lg")

# Define the custom function to match any token

# Define the target patterns

target_rule = {
    "Necrosis": [
        [{"LOWER": "necrosis"}]
    ],
    "Mitosis": [
        [{"LOWER": {"in": ["mitosis", "mitosens", "mitose"]}}]
    ],
    "Grade": [
        [{"LOWER": "risk"}, {"LOWER": "stratification"}],
        [{"LOWER": {"in": ["grade", "grading"]}}]
    ],
}

# You can use this to save target rule
#serializable_target_rule = [(key, value) for key, value in target_rule.items()]
#with open("pipe.json", "w") as json_file:
#    json.dump(serializable_target_rule, json_file, indent=4)

# Initialize the Matcher with the spaCy pipeline
matcher = Matcher(nlp.vocab)

# Add the patterns to the matcher
for rule_name, rule in target_rule.items():
    matcher.add(rule_name, rule)

# Example text
text = """
The tumor reaches into the inked outer surface. The gastric resection surface is free. Risk stratification: low.
The tumor reaches into the inked outer surface. The gastric resection surface is free. Grading low.
The tumor reaches into the inked outer surface. The gastric resection surface is free. Grade 2.
The grading is low for this tumor.
Determined grade: 3.
"""

text = """
A1: Skin biopsy reaching into the dermis coated with multilayered horny squamous epithelium with a thick hyperkeratotic plug with presence of bacteria and serum. The epidermis shows reactive changes. The entire dermis is taken by an atypical epithelioid proliferation with splitting vascular spaces between the tumour cells. There is infiltrational growth mode between the pre-existent collagen. The tumor cells show enlarged polymorphic hyperchromatic nuclei with sometimes prominent nucleolus.
A2 and A3: display a similar image. Skin biopsy reaching into the subcutaneous fat coated with multilayered horny squamous epithelium with basket weave ortho keratosis. In particular, in the superficial dermis there are teleangiectatic dilated capillaries which are partly split between the collagen and are coated with partly non atypical endothelial and partly there is tufting in lumen with also loosely located atypical cells with enlarged polymorphic hyperchromatic nuclei. Partly an anastomizing aspect. Around hemosiderin pigment.. Immune histochemistry (everywhere performed and sent): A1: Lesional cells are positive for CD31 and partly for CD117 and negative for CKAE1/AE3, S100, Melan-A, CD56, CD3, CD20, CD138, CD34, p16, CD1a, factor 13A and MPO.
The tumor cells are strongly positive for ERG (directed at the Maasstad hospital).
"""

text = """
Immunohistochemistry: tumour cells are positive for DOG-1 and CD117
"""

text = """
Around some deposition of haemosiderin pigment
"""

# Process the text with spaCy
doc = nlp(text)

print(doc.ents)

displacy.serve(doc, style="dep")

# Apply the matcher to the processed text
matches = matcher(doc)

# Extract the matched phrases
for match_id, start, end in matches:
    span = doc[start:end]
    print(f"Matched phrase: {span.text}")
