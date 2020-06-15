import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class ExtDataset(Dataset):
    def __init__(self, tokenizer, path):

        self.tokenizer = tokenizer

        data = glob.glob(os.path.join(rf'{path}', r'**\*.txt'), recursive=True)

        text = []  # variable to store text
        all_inputs = []
        all_target = []
        all_cls_mask = []
        all_pad_mask = []
        all_segment_ids = []

        for file in tqdm(data, total=len(data)):
            with open(file, 'r', encoding='utf8') as f:
                lines = f.read()
                line_split = lines.split('\n')
                text.append(line_split[0])  # First line is the test
                label = line_split[1].split(',')  # Second line is labels separated by a comma

                x = tokenizer.encode(line_split[0], max_length=512, pad_to_max_length=True)
                pad_mask = self.get_att_mask(x)  # Attention mask
                cls_mask = self.get_cls_mask(x)  # Cls mask
                seg_id = self.get_segment_id(x)  # Segment id

                cls_idx = np.argwhere(cls_mask).squeeze()  # get the index of the CLS token in the sequence
                # the target has the the same shape as the input

                # most of the element of the target will be zero except in positions where a CLS is a summary
                target = np.zeros_like(x)
                try:
                    for j, id in enumerate(cls_idx):
                        target[id] = 1.0 if str(j) in label else 0.0  # if position is in summary id
                except TypeError: # Happen when there is only one sentence in the text
                    target[0] = 1.0

                all_inputs.append(x)
                all_target.append(target)
                all_cls_mask.append(cls_mask)
                all_pad_mask.append(pad_mask)
                all_segment_ids.append(seg_id)
                # Appending

        self.text = text

        self.x = torch.LongTensor(all_inputs)  # Default format for BERT input
        self.y = torch.FloatTensor(all_target)  # Float since we will use BCE loss
        self.cls_mask = torch.FloatTensor(all_cls_mask)  # Float by default
        self.pad_mask = torch.FloatTensor(all_pad_mask)  # Float by default
        self.segment = torch.LongTensor(all_segment_ids)  # Long because input of an embedding layer

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.cls_mask[index], self.pad_mask[index], self.segment[index], self.text[
            index]

    def get_cls_mask(self, token_ids):  # function to get cls masks
        return list(map(lambda a: 1 if a in [self.tokenizer.cls_token_id] else 0, token_ids))

    def get_att_mask(self, token_ids):  # function to get att masks
        return list(map(lambda a: 0 if a in [self.tokenizer.pad_token_id] else 1, token_ids))

    def get_segment_id(self, token_ids):  # function to get segment id
        flag = 0
        segment = []
        for id_ in token_ids:
            segment.append(flag % 2)
            if id_ is self.tokenizer.sep_token_id:
                flag += 1
        return segment
