from operator import itemgetter

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import ExtDataset
from model import BaselineEXT

tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')

mod = BaselineEXT(num_emb=tokenizer.vocab_size,
                  emb_dim=64,
                  n_head=4,
                  num_layers=1)

mod.load_state_dict(torch.load(r'trained_model\model_step_50.pt'))

mod.eval()

valid = ExtDataset(tokenizer, path=r'sample_data\valid', train=False)
data_loader = DataLoader(valid, batch_size=1, shuffle=False)

texts = []
summaries = []
summaries_id = []
for i, (x, y, cls_mask, att_mask, seg, text) in enumerate(data_loader):
    y_hat, _, _ = mod(x, y, cls_mask, att_mask, seg)

    cls_ids = np.argwhere(cls_mask.numpy().squeeze()).squeeze()  # get cls_id
    masked_y_hat = y_hat * cls_mask

    top_sentences = torch.topk((y_hat * cls_mask).squeeze(), k=3)[1]
    top_sentences = top_sentences.numpy().squeeze()

    true_sum = np.argwhere(y.numpy().squeeze())
    summary_id = []
    for j, idx in enumerate(cls_ids):
        summary_id.append(j) if idx in top_sentences else 0

    text = ' '.join(list(text)).split('[SEP][CLS]')

    text_summary = itemgetter(*summary_id)(text)

    texts.append(text)
    summaries.append(text_summary)
    summaries_id.append(summary_id)

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
