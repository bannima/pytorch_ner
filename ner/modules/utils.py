
import argparse
import datetime
import json
import os.path
import time
import joblib
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset
from sklearn.preprocessing import LabelBinarizer,MultiLabelBinarizer

def parse_parmas():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=False, default=32)
    parser.add_argument('--lr', type=float, required=False, default=2e-3)
    parser.add_argument('--epoch', type=int, required=False, default=5)
    parser.add_argument('--criterion', type=str, required=False, default='SeqWiseCrossEntropyLoss')
    parser.add_argument('--metrics', type=str, required=False, default='multiclass')
    parser.add_argument('--save_model', type=bool, required=False, default=False)
    parser.add_argument('--save_test_preds', type=bool, required=False, default=True)
    parser.add_argument('--save_val_preds', type=bool, required=False, default=True)
    return parser.parse_args()

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def which_day():
    return time.strftime('%Y-%m-%d', time.localtime(time.time()))


def current_time():
    return time.strftime("%Y%m%d_%H%M", time.localtime(time.time()))


def save_to_json(datas, filepath, type='w'):
    with open(filepath, type, encoding='utf-8') as fout:
        for data in datas:
            json.dump(data, fout, ensure_ascii=False)
            fout.write('\n')


def read_from_json(filepath):
    if not os.path.exists(filepath):
        raise RuntimeError(" File not exists for {}".format(filepath))
    dataframe = []
    with open(filepath, 'r', encoding='utf-8') as fread:
        for line in tqdm(fread.readlines(), desc=' read json file', unit='line'):
            dataframe.append(json.loads(line.strip('\n')))
    return dataframe


def flatten(lists):
    ''' flatten nested lists-like object '''
    for x in lists:
        if hasattr(x, '__iter__') and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x

def load_word_vectors(vec_path,vocab_path=None):
    """ load the word vectors """
    if vocab_path is not None:
        with open(vocab_path,'r') as fp:
            vocab = {word:idx for idx,word in enumerate(fp)}
        return np.load(vec_path,allow_pickle=True),vocab
    else:
        return np.load(vec_path,allow_pickle=True)

def get_label_binarizer(lb_path,labels=None):
    '''
    get the label binarizer
    :param lb_path:
    :param labels:
    :return:
    '''
    if os.path.exists(lb_path):
        return joblib.load(lb_path)
    label_binarizer = MultiLabelBinarizer(sparse_output=True)
    label_binarizer.fit(labels)
    joblib.dump(label_binarizer,lb_path)
    return label_binarizer


def convert_to_bert_batch_data(data, tokenizer, max_length):
    '''convert raw input and labels to bert dataset'''
    input_ids = []
    attention_masks = []
    token_type_ids = []

    for row in data:
        encoded_dict = tokenizer.encode_plus(
            row,max_length = max_length,pad_to_max_length=True,\
            return_attention_mask=True,return_tensors='pt',truncation=True
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        # bert-type model has token_type_ids
        try:
            token_type_ids.append(encoded_dict['token_type_ids'])
        except:
            pass
    #convert list to tensor
    input_ids = torch.cat(input_ids,dim=0)
    attention_masks = torch.cat(attention_masks,dim=0)

    if len(token_type_ids)!=0:
        token_type_ids = torch.cat(token_type_ids,dim=0)
        return (input_ids,attention_masks,token_type_ids)
    else:
        return (input_ids,attention_masks)













