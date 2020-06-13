import argparse

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from transformers import AutoTokenizer

from dataset import ExtDataset
from model import BertExtSum, BaselineFFN


def train(model, optimizer, train_loader, valid_loader=None, evaluate_every=50, max_step=10):
    train_losses = []
    train_accuracy = []

    model.train()
    device = next(model.parameters()).device

    step = 0

    while True:
        if step > max_step:
            break
        for x, y, cls_mask, att_mask, _, _ in tqdm(train_loader, leave=False):
            y_hat, masked_loss, masked_acc = model(x.to(device), y.to(device), cls_mask.to(device), att_mask.to(device))
            optimizer.zero_grad()
            masked_loss.backward()
            optimizer.step()
            train_losses.append(masked_loss.item())
            train_accuracy.append(masked_acc.item())

            step += 1

            print(f'train mean loss = {np.mean(train_losses[-1])}')
            print(f'train mean acc = {np.mean(train_accuracy[-1])}')

            if valid_loader:
                if step % evaluate_every == 0:
                    model.eval()
                    val_losses = []
                    val_accuracy = []
                    print('validation...')
                    with torch.no_grad():
                        for x_val, y_val, cls_mask_val, att_mask_val, _, _ in tqdm(valid_loader, leave=False):
                            y_hat, masked_loss, masked_acc = model(x_val.to(device), y_val.to(device),
                                                                   cls_mask_val.to(device),
                                                                   att_mask_val.to(device))
                            val_losses.append(masked_loss.item())
                            val_accuracy.append(masked_acc.item())

                    model.train()

                    print(f'val mean loss = {np.mean(val_losses)}')
                    print(f'val mean acc = {np.mean(val_accuracy)}')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model", help='Name of the model',
                        default='camembert-base')
    parser.add_argument("--device", help='Name of the model',
                        default='camembert-base')
    parser.add_argument("--train_path", help='Path of the training data folder',
                        default=r'french_data\comb')
    parser.add_argument("--valid_path", help='Path of the validation data folder',
                        default=r'french_data\valid')
    parser.add_argument("--max_step", help='Maximum number of step',
                        default=60)
    parser.add_argument("--evaluate_every", help='',
                        default=20)
    parser.add_argument("--lr", help='Learning rate',
                        default=1e-3)
    parser.add_argument("--train_Bs", help='Training batch size',
                        default=4)
    parser.add_argument("--val_Bs", help='Validation batch_size',
                        default=16)
    parser.add_argument("--add_transformer_layers", help='Validation batch_size',
                        default=False)
    parser.add_argument("--num_layers", help='Number of transformer layers',
                        default=1)
    parser.add_argument("--n_head", help='Number of head of the transformer layers',
                        default=4)
    parser.add_argument("--baseline", help='If run baseline',
                        default=False)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    data = ExtDataset(tokenizer, path=args.train_path)
    valid = ExtDataset(tokenizer, path=args.valid_path)

    loader = DataLoader(data, batch_size=int(args.train_Bs))
    v_loader = DataLoader(valid, batch_size=int(args.val_Bs))

    if args.baseline:
        model = BaselineFFN(bert_model=args.bert_model)
    else:
        model = BertExtSum(bert_model=args.bert_model,
                           add_transformer_layers=args.add_transformer_layers,
                           num_layers=int(args.num_layers), n_head=int(args.n_head))

    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    train(model, opt, loader, max_step=args.max_step, valid_loader=v_loader,
          evaluate_every=int(args.evaluate_every))
