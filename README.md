# NER-System

A PyTorch-based NER-System repository for exploring various NER datasets, training deep learning models, and sharing them with the community.

### Libraries:
- Python-3.9
- torch-1.11
- transformers-4.12
- datasets-2.3
- seqeval-1.2

### Dataset : OntoNotes v5

- Data Acquisition - [download_ontonotes.ipynb](download_ontonotes.ipynb)

    | Dataset | Examples |
    | --- | --- | 
    | Training | 75187 | 
    | Testing | 9479 |  

    | **Entity Tag**                        | **Meaning** |
    |---------------------------------|-----------|
    | CARDINAL    | cardinal value | 
    | DATE         | date value | 
    | EVENT         | event name | 
    | FAC         | building name | 
    | GPE         | geo-political entity | 
    | LANGUAGE         | language name | 
    | LAW         | law name | 
    | LOC         | location name | 
    | MONEY         | money name | 
    | NORP         | affiliation | 
    | ORDINAL         | ordinal value | 
    | ORG         | organization name | 
    | PERCENT         | percent value | 
    | PERSON         | person name | 
    | PRODUCT         | product name | 
    | QUANTITY         | quantity value | 
    | TIME         | time value | 
    | WORK_OF_ART         | name of work of art | 

- EDA (Exploratory Data Analysis) - [ner_eda.ipynb](ner_eda.ipynb)
- Model Training and Evaluation : [train_ner.py](train_ner.py)

    | Model Name | Precision | Recall | F1 Score |
    | --- | --- | --- | --- |
    | roberta-base <br> ([ner-roberta-base-ontonotesv5-englishv4](https://huggingface.co/djagatiya/ner-roberta-base-ontonotesv5-englishv4)) | 88.88 | 90.69 | 89.78 |
    | distilbert-base-uncased <br> ([ner-distilbert-base-uncased-ontonotesv5-englishv4](https://huggingface.co/djagatiya/ner-distilbert-base-uncased-ontonotesv5-englishv4)) | 84.60 | 86.47 | 85.53 |


- Inference using pipeline: [infer_pipeline.ipynb](infer_pipeline.ipynb)