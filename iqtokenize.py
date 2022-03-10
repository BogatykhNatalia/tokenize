#!/usr/bin/env python
# coding: utf-8

from transformers import AutoTokenizer
import torch
from utils import save_json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-i','--input',
                    type=str,
                    default='This is an example sentence')
parser.add_argument('-o','--output',
                    type=str,
                    default='tokenized_text.json')
parser.add_argument('--model_name',
                    type=str,
                    default='sentence-transformers/distiluse-base-multilingual-cased-v2')
args = parser.parse_args()

model_name = args.model_name
text = args.input
output_name = args.output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

stop_symbols = '/\\;-+=@#%`“«»<>—–--``''\'[]{}*•\n'
table = str.maketrans(dict.fromkeys(stop_symbols,' '))

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize sentences
encoded_input = tokenizer(
                    ' '.join(text.translate(table).split(' ')),
                    padding=True,
                    truncation=True,
                    return_tensors='pt').to(device)

save_json(output_name,{'text':text,'tokens':encoded_input['input_ids'].cpu().numpy(),'attention_mask':encoded_input['attention_mask'].cpu().numpy()})