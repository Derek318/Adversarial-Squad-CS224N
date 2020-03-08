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

from sklearn.model_selection import train_test_split




# 60 20 20 split

def pre_process():
# Process training set and use it to decide on the word/character vocabularies
    word_counter, char_counter = Counter(), Counter()

   
   #This takes args.train_file
   # all examples = [dicts]
   # all_eval  = {id -> dict}
   
    #POTENTIAL BUG: MAY BE BAD TO DO WORD COUNTER ON "ENTIRE" DATASET rather than just train like in orig setup.py
    all_examples, all_eval = setup.process_file("./adversarial_dataset.json", "all", word_counter, char_counter)
    all_indices = list(map(lambda e: e['id'], all_examples))
    
#    import pdb; pdb.set_trace()

#    print(all_examples[0]["context_tokens"], all_examples[0]["ques_tokens"])
#    print(all_examples[1]["context_tokens"], all_examples[1]["ques_tokens"])
#    print(type(all_examples))
#    print(type(all_eval))
    # indices are from  0 to 3559 (3560 questions total)

    
    # 2136 total questions and answers in train
    # 712 questions + answers in dev
    # 712 questions + answers in test
    train_examples, residual_examples = train_test_split(all_examples, test_size=0.4)
    dev_examples, test_examples = train_test_split(residual_examples, test_size=0.5)
    
    train_eval = {str(e['id']) : all_eval[str(e['id'])] for e in train_examples}
    dev_eval = {str(e['id']) : all_eval[str(e['id'])] for e in dev_examples}
    test_eval = {str(e['id']) : all_eval[str(e['id'])] for e in test_examples}


    # IMPORTANT: Ensure that we do not split corresponding question and answers into different datasets
    assert set([str(e['id']) for e in train_examples]) == set(train_eval.keys())
    assert set([str(e['id']) for e in dev_examples]) == set(dev_eval.keys())
    assert set([str(e['id']) for e in test_examples]) == set(test_eval.keys())

    # TODO: Call the rest of the setup.py to get the .npz files
    # TODO: Once we have the .npz, we can call test on the adversarial data
    # TODO: Re-train BiDAF on adversarial dataset
    # TODO: Data augmentation
    # TODO: Auxiliary Model to predict sentence relevancy
    
    
    # ========= FROM SETUP.PY =========== #
    # Need to create the .npz, .json files for dev, test, and train
    # this is desired structure for training/testing
    
    args = get_setup_args()
    
    
    # Setup glove path for adversarial dataset
    glove_dir = setup.url_to_data_path(args.glove_url.replace('.zip', ''))
    glove_ext = f'.txt' if glove_dir.endswith('d') else f'.{args.glove_dim}d.txt'
    args.glove_file = os.path.join(glove_dir, os.path.basename(glove_dir) + glove_ext)
    
    
    # Setup word, char embeddings for adversarial data
    word_emb_mat, word2idx_dict = setup.get_embedding(word_counter, 'word', emb_file=args.glove_file, vec_size=args.glove_dim, num_vectors=args.glove_num_vecs)
    char_emb_mat, char2idx_dict = setup.get_embedding(char_counter, 'char', emb_file=None, vec_size=args.char_dim)
      
      
    #args.train_record_file is the .npz file path that we want to save stuff to
    setup.build_features(args, train_examples, "train", "./adv_data/train.npz", word2idx_dict, char2idx_dict)
    dev_meta = setup.build_features(args, dev_examples, "dev", "./adv_data/dev.npz", word2idx_dict, char2idx_dict)
      
    # True by default
    if args.include_test_examples:
        # Step done above
#        test_examples, test_eval = process_file("./adversarial_dataset/test-v2.0.json", "adv test", word_counter, char_counter)
        setup.save("./adv_data/test_eval.json", test_eval, message="adv test eval")
        test_meta = setup.build_features(args, test_examples, "adv test", "./adv_data/test.npz", word2idx_dict, char2idx_dict, is_test=True)
        setup.save("./adv_data/test_meta.json", test_meta, message="adv test meta")

    setup.save("./adv_data/word_emb.json", word_emb_mat, message="word embedding")
    setup.save("./adv_data/char_emb.json", char_emb_mat, message="char embedding")
    setup.save("./adv_data/train_eval.json", train_eval, message="adv train eval")
    setup.save("./adv_data/dev_eval.json", dev_val, message="adv dev eval")
    setup.save("./adv_data/word2idx.json", word2idx_dict, message="word dictionary")
    setup.save("./adv_data/char2idx.json", char2idx_dict, message="char dictionary")
    setup.save("./adv_data/dev_meta.json", dev_meta, message="adv dev meta")
    
    
    # ========= FROM SETUP.PY =========== #
            
            
def test_baseline():
    pass
    

    
if __name__ == '__main__':
    pre_process()
    
    test_baseline()
    
