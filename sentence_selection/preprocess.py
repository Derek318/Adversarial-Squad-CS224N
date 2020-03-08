import logging
import os
import queue
import random
import re
from args import get_setup_args
import shutil
import string
import setup
import torch
import torch.nn.functional as F
import torch.utils.data as data
from collections import Counter
import tqdm
import numpy as np
import ujson as json
import spacy
import json

# Sentence encodings
from bert_sent_encoding import bert_sent_encoding


def pre_process():
# Read in train test, dev jsons
# Get each paragraph

# TRAIN
# Split into {question_id -> ("question_string", ["sent1_str", "sent2_str", "sent3_str", ...]))}
# Split into {qid -> (q_embed,[sent1_embed, sent2_embed, sent3_embed, ....])}

#DEV + TEST
# Do same for dev + test

#populate three dictionaries

    datasets = ["dev","test","train"]
    v_str = "-v2.0.json"

    for ds in datasets:
        filename = ds + v_str
        with open(filename) as f:
            json_file = json.load(f)
            # outermost is a keys list of [version, data]
            data = json_file["data"]
            # Data is a list of dicts
            ex_entry = data[0]
            
            # Each entry is a dict of keys [title, paragraphs]
            print(ex_entry['title']) #Title example is normans
            #Paragraphs is a list of dicts
#            print(ex_entry['paragraphs'])
            ex_paragraphs = ex_entry['paragraphs']
            print(ex_paragraphs[0])
            

#            bse = bert_sent_encoding(model_path='bert_sent_encoding/model/chinese_L-12_H-768_A-12')
#            bse = bert_sent_encoding(model_path='bert-base-cased')
#            vector = bse.get_vector('Hello', word_vector=False, layer=-1)   # 3rd line 1. get vector of string
#            exit()
            vectors = bse.get_vector(['sent2', 'sent3'], word_vector=False, layer=-1)  # 4th line 2. get vector list of strings
#            bse.write_txt2vector(input_file, output_file, word_vector=False, layer=-1)   # 5th line 3. get and write vectors of strings


if __name__ == '__main__':
    pre_process()

