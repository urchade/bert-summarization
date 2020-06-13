import os

from torch import nn
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import numpy as np
from dataset import ExtDataset
from model import BertExtSum, BaselineFFN
from operator import itemgetter

# @torch.no_grad()
# def translate(data_path=r'french_data\valid', target_file=r'french_data\target_file'):
#     model = BertExtSum(bert_model='camembert-base',
#                        add_transformer_layers=False,
#                        n_head=0,
#                        num_layers=0)
#     model.eval()
#     # model.load_state_dict(args.state_dict["model"])
#     tokenizer = AutoTokenizer.from_pretrained('camembert-base')
#
#     valid = ExtDataset(tokenizer, path=data_path)
#     loader = DataLoader(valid, batch_size=1, shuffle=False)
#
#     for i, (x, y, cls_mask, att_mask, _, text) in enumerate(loader):
#         y_hat, _, _ = model(x, y, cls_mask, att_mask)
#         cls_ids = np.argwhere(cls_mask.numpy().squeeze()).squeeze()
#         top_sentences = torch.topk((y_hat * cls_mask).squeeze(), k=2)[1]
#         top_sentences = top_sentences.numpy().squeeze()
#
#         summary_id = []
#         for j, idx in enumerate(cls_ids):
#             summary_id.append(j) if idx in top_sentences else 0
#
#         text = ' '.join(list(text)).split('</s><s>', )
#
#         text_summary = ''.join(itemgetter(*summary_id)(text)).strip()
#
#         with open(target_file, 'a', encoding='utf8') as f:
#             f.write(text_summary)
#             f.write('\n')

model = BaselineFFN(bert_model='camembert-base',
                    add_transformer_layers=False,
                    n_head=0,
                    num_layers=0)

model.load_state_dict(torch.load('model.pt'))

model.eval()
# model.load_state_dict(args.state_dict["model"])
tokenizer = AutoTokenizer.from_pretrained('camembert-base')

valid = ExtDataset(tokenizer, path=r'french_data\valid')
loader = DataLoader(valid, batch_size=1, shuffle=False)

for i, (x, y, cls_mask, att_mask, _, text) in enumerate(loader):
    y_hat, _, _ = model(x, y, cls_mask, att_mask)
    cls_ids = np.argwhere(cls_mask.numpy().squeeze()).squeeze()
    top_sentences = torch.topk((y_hat * cls_mask).squeeze(), k=3)[1]
    top_sentences = top_sentences.numpy().squeeze()

    summary_id = []
    for j, idx in enumerate(cls_ids):
        summary_id.append(j) if idx in top_sentences else 0

    text = ' '.join(list(text)).split('</s><s>', )

    text_summary = ''.join(itemgetter(*summary_id)(text)).strip()

    import numpy as np

    if np.random.uniform(0, 1) > 0.95:
        break
