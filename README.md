# MedicalRecordNLP

## LLM



## NLP 

We have made an NLP pipeline which extract several entities from semi-structured reports.

This NLP workflow consists of several things:
- **Translation** of Dutch to English using a language transformer from [Hugging Face](https://huggingface.co/Helsinki-NLP/opus-mt-nl-en).
- **Named entities recognition (NER)** using [scispacy](https://allenai.github.io/scispacy/), specifically the NER model trained on the BIONLP13CG corpus (en_ner_bionlp13cg_md)
- **EntityLinker** of NER against the Unified Medical Language System (UMLS) to both further identify NERs specific to our problem and standardize the naming for downstream use.
- **Handcrafted matcher** to find NER specific to our problem not picked up by specialized NER model.
- **Dependency analysis** to connect to NER.

Now shortly a description of what all files are:
- **translate.py** is pretty self explanatory. It takes a rapport and translates it using the previous described language transformer.
- **report.py** contains the 'main' class function for standardized reporting. This is usefull for everyone, as it contains a lot of helper functions for all the tasks named above not specifically for the reports we use it for.
- **pathology.py** is a more specialized script that builds upon the standardized reporting class specifically made to extract information from pathology reports, with a focus on extracting NER (and their dependencies) for phenotype (cancer), grade, genetic variations or immunohistochemistry staining, mitosis, necrosis, and tumor location. Note, that we tested this for soft tissue tumors, but it should also be applicable for other tumor types.
- **pipe.json** is a handcrafted matcher, which can be used in standardized reporting to find NERs. For our use case we identify grade, mitosis and necrosis, but you can adjust this file as you wish.
- **regex_pattern.json** is used to extract info more precisely from different parts of the report. Thereby, we can use the semi-structured format of the report in our advantage when searching for best matching NERs. For example, we look for phenotype and grade in the conclusion of the report.

Note before continuing, scispacy is a bit weird as newer versions (compatible with spacy>3.x) provide worse results than older versions. This is a known [issue](https://github.com/allenai/scispacy/issues/342#issuecomment-804993320). Therefore, I use a downgraded version of scispacy 0.3.0 and Python 3.7.4. Finally, also make sure you download the matching [BIONLP13CG model](https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.3.0/en_ner_bionlp13cg_md-0.3.0.tar.gz) matching scispacy 0.3.0.

In order to analyze reports, first, translate the reports from dutch to english:
```sh
python translate.py -i ../data/Verslagen_202310112035.csv
```

Next, you we analyze the results using pathology.py. Notice that we also provide the regex patterns and matcher pipe.
```sh
python pathology.py -i ../data/pathology_translated.csv -o ../data/analyzed_reports.jsonl -r ../resources/regex_pattern.json -p ../resources/pipe.json
```

The output of the analyzes is in the jsonl format, which is directly compatible with Prodigy.

### Prodigy

We use prodigy to check and correct the analyzed reports in NLP, so we can adjust where needed. Some notes beforehand, I would generally recommend to setup another virtual environment with a newer version of Python > 3.9 to run Prodigy, as otherwise you are missing out on a lot of newer nice functionalities.

First, we read the analyzed report from NLP into the prodigy SQLite database (change pathology for your desired name):
```sh
prodigy db-in pathology ../data/analyzed_reports.jsonl
```

Next, we can start the prodigy server with the loaded database (Note, you should provide the labels and span-labels used in your dataset):
```sh
prodigy rel.manual pathology_corrected blank:en dataset:pathology --loader pages:jsonl --label CANCER,GRADE,ORGAN,CELL,MITOSIS,NECROSIS,GENE_OR_GENE_PRODUCT,GRANULAR_DISEASE,SUSPECTED_CANCER,OTHERDISEASE --span-label CANCER,GRADE,ORGAN,CELL,MITOSIS,NECROSIS,GENE_OR_GENE_PRODUCT,GRANULAR_DISEASE,SUSPECTED_CANCER,OTHERDISEASE
```

Note that `--loader pages:jsonl` is used above, this is optional for loading multiple reports. I use this since I have multiple pathology reports for each patient.

Prodigy keeps track of your annotation process and saves everything in the SQLite database. Finally, you can read either directly from this [SQLite database](https://prodi.gy/docs/api-database) or you extract the database to the jsonl format used also initally as input:
```sh
prodigy db-out pathology_corrected >../corrected_reports.jsonl
```

For more information checkout [Prodigy](https://prodi.gy/docs). 