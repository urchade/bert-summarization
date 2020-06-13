from torch import nn
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import numpy as np
from dataset import ExtDataset
from model import BertExtSum
from operator import itemgetter

model = BertExtSum(bert_model='camembert-base', add_transformer_layers=False, n_head=1, num_layers=1)
tokenizer = AutoTokenizer.from_pretrained('camembert-base')

valid = ExtDataset(tokenizer, path=r'french_data\valid')
loader = DataLoader(valid, batch_size=1)

for x, y, cls_mask, att_mask, _, text in loader:
    break

y_hat, _, _ = model(x, y, cls_mask, att_mask)

cls_ids = np.argwhere(cls_mask.numpy().squeeze()).squeeze()

top_sents = torch.topk((y_hat * cls_mask).squeeze(), k=2)[1]

top_sents = top_sents.numpy().squeeze()

summary_id = []
for i, id in enumerate(cls_ids):
    summary_id.append(i) if id in top_sents else 0

text = ' '.join(list(text)).split('</s><s>')

text_summary = ''.join(itemgetter(*summary_id)(text)).strip()
