# %% Import libraries
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from seqeval.metrics import classification_report 

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast, DataCollatorForTokenClassification, \
    AutoModelForTokenClassification, Trainer, TrainingArguments, DistilBertForTokenClassification
import datasets

import logging
import logging.config

import json

from my_logger import set_logger

import time
import os

import random

# %% Config

class Config:
    # model_path = 'models/roberta-base_1656662418.0944197/checkpoint-9400'
    # load_fine_tunned = False

    model_name = 'distilbert-base-uncased'
    out_dir = f"./models/{model_name}_{time.time()}"
    data_path = Path("data/ontonotes")
    train_data_limit = 5000
    test_data_limit = 1000
    batch_size = 16
    random_seed = 1
    logging_steps_per_epochs = 10
    do_train = True
    do_eval = True
    
    training_args = dict(
        output_dir = out_dir,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        save_total_limit = 2,
        num_train_epochs = 3,
        seed = random_seed,
        load_best_model_at_end=True,
        evaluation_strategy="epoch",
        save_strategy='epoch',
        learning_rate=2e-5,
        weight_decay=0.01,
    )


# if Config.load_fine_tunned:
#     Config.model_name = Config.model_path    

os.mkdir(Config.out_dir)

set_logger(f"{Config.out_dir}/run.log")


_logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(Config.random_seed)


# %% data loading utility functions
def format_data(lines):
    """
        Sepearate line into two `words` and `tags` list.
    """
    word_ls, tag_ls = [], []
    for _l in lines:
        _word, _tag = _l.strip().split("\t")
        word_ls.append(_word)
        tag_ls.append(_tag)
    return word_ls, tag_ls

def load_data(path, features):
    """
    Read conll BIO format and convert to dataset object
    """
    with open(path, 'r') as _file:
        all_data = []
        for i, (x, y) in enumerate(itertools.groupby(_file, lambda x : x == '\n')):
            if x: continue
            y = format_data(y)
            all_data.append(y)
        # TODO: Not good way to create dataset from combination of dataframe with `from_dict` method.
        df = pd.DataFrame(all_data, columns=["words", "ner_tags"])
        data = datasets.Dataset.from_dict({
            "id" : df.index, 
            "words" : df.words.values, 
            "ner_tags" : df.ner_tags.values
        }, features=features)
        return data 
# %%

with open(Config.data_path / 'labels.json', mode='r', encoding='utf-8') as _file:
    label_list =  json.load(_file)['names']

_logger.info(f"Label List:{label_list}")

features = datasets.Features({
    'id' : datasets.Value(dtype='int32'),
    "words" : datasets.Sequence(feature=datasets.Value(dtype='string')),
    "ner_tags" : datasets.Sequence(feature=datasets.features.ClassLabel(num_classes=len(label_list), names=label_list))
})

# %%
train_data = load_data(Config.data_path / "train.conll", features)
_logger.info(f"{train_data}")

test_data = load_data(Config.data_path / "test.conll", features)
_logger.info(f"{test_data}")


if Config.train_data_limit is not None:
    train_data = train_data.select(list(range(Config.train_data_limit)))
    _logger.info(train_data)

if Config.test_data_limit is not None:
    test_data = test_data.select(list(range(Config.test_data_limit)))
    _logger.info(test_data)
# %%
tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

# %% [DEMO]

examples = train_data[:5]['words']
encoded = tokenizer(examples, is_split_into_words=True, padding=True)
_logger.info(encoded)

for i in range(5):
    _logger.info(tokenizer.convert_ids_to_tokens(encoded['input_ids'][i]))

_logger.info("-------------")
_logger.info(tokenizer.convert_ids_to_tokens(encoded['input_ids'][1]))
_logger.info(encoded.word_ids(1))

# %%

# %%
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["words"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None: # [CLS], [SEP] 
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100) # subwords of current word
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


print(tokenize_and_align_labels(train_data[:5]))

# %% 
# Utilize `map` function of datasets api to tokenize and realign input.  
train_data = train_data.map(tokenize_and_align_labels, batched=True)
_logger.info(train_data[:5])

test_data = test_data.map(tokenize_and_align_labels, batched=True)

# %%

data_collector = DataCollatorForTokenClassification(tokenizer)


# %%
# model(
#     input_ids=torch.tensor(encoded['input_ids'], dtype=torch.int), 
#     attention_mask=torch.tensor(encoded['attention_mask'], dtype=torch.int)
# )

# %%
seqeval_metric = datasets.load_metric('seqeval')

def get_true_and_prediction(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]  
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return true_predictions, true_labels

def compute_metrics(p):
    predictions, labels = get_true_and_prediction(p)
    results = seqeval_metric.compute(predictions=predictions, references=labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# %%
# DistilBertForTokenClassification

# if 'model' in globals():
#     _logger.info("Deleting previous model.")
#     del model

label_str2int = features['ner_tags'].feature._str2int
label_int2str = {v:k for k,v in label_str2int.items()}

model = AutoModelForTokenClassification.from_pretrained(Config.model_name, **{
    "num_labels" : len(label_list),
    "id2label" : label_int2str,
    "label2id" : label_str2int
})

_logger.info(f"{model}")

# %%


Config.training_args["logging_steps"] = (len(train_data) / Config.batch_size) // Config.logging_steps_per_epochs

_logger.info(f"CONFIGS:{json.dumps(Config.training_args, indent=4)}")

training_args = TrainingArguments(**Config.training_args)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=tokenizer,
    data_collator=data_collector,
    compute_metrics=compute_metrics
)

if Config.do_train:
    _logger.info("[[ MODEL TRAINING ]]")
    train_result = trainer.train()

# %%
if Config.do_eval:
    _logger.info("[[ MODEL EVALUATION ]]")
    trainer.evaluate()
# %%

for _h in trainer.state.log_history:
    _logger.info(_h)


p = trainer.predict(test_data)
prediction, true = get_true_and_prediction(p[:2])

report = classification_report(true, prediction)

_logger.info(f"{report}")
# %%
