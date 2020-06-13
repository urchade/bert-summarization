import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class ExtDataset(Dataset):
    def __init__(self, tokenizer, path):

        def get_cls_mask(token_ids):  # function to get cls masks
            return list(map(lambda a: 1 if a in [tokenizer.cls_token_id] else 0, token_ids))

        def get_att_mask(token_ids):  # function to get att masks
            return list(map(lambda a: 0 if a in [tokenizer.pad_token_id] else 1, token_ids))

        def get_segment_id(token_ids):
            flag = 0
            segment = []
            for id_ in token_ids:
                segment.append(flag % 2)
                if id_ is tokenizer.sep_token_id:
                    flag += 1
            return segment

        data = glob.glob(os.path.join(rf'{path}', r'**\*.txt'), recursive=True)

        text = []  # variable to store text
        label = []  # variable to store label list
        for file in data:
            with open(file, 'r', encoding='utf8') as f:
                lines = f.read()
                line_split = lines.split('\n')
                text.append(line_split[0])  # First line is the test
                label.append(line_split[1].split(','))  # Second line is labels separated by a comma

        self.text = text
        # Some variables to store inputs and masks
        all_inputs = []
        all_target = []
        all_cls_mask = []
        all_pad_mask = []
        all_segment_ids = []

        for i, (x, y) in enumerate(zip(text, label)):
            x = tokenizer.encode(x, max_length=512, pad_to_max_length=True)  # encoding the text
            pad_mask = get_att_mask(x)  # Attention mask
            cls_mask = get_cls_mask(x)  # Cls mask
            seg_id = get_segment_id(x)  # Segment id

            cls_idx = np.argwhere(cls_mask).squeeze()  # get the index of the CLS token in the sequence
            # the target has the the same shape as the input

            # most of the element of the target will be zero except in positions where a CLS is a summary
            target = np.zeros_like(x)
            for j, id in enumerate(cls_idx):
                target[id] = 1.0 if str(j) in y else 0.0  # if position is in summary id

            all_inputs.append(x)
            all_target.append(target)
            all_cls_mask.append(cls_mask)
            all_pad_mask.append(pad_mask)
            all_segment_ids.append(seg_id)
            # Appending

        self.x = torch.LongTensor(all_inputs)  # Default format for BERT input
        self.y = torch.FloatTensor(all_target)  # Float since we will use BCE loss
        self.cls_mask = torch.FloatTensor(all_cls_mask)  # Float by default
        self.pad_mask = torch.FloatTensor(all_pad_mask)  # Float by default
        self.segment = torch.LongTensor(all_segment_ids)  # Long because input of an embedding layer

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.y[item], self.cls_mask[item], self.pad_mask[item], self.segment[item], self.text[item]
