import argparse

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from transformers import AutoTokenizer

from dataset import ExtDataset
from model import BertExtSum, BaselineEXT


def train(network, optimizer, train_loader, valid_loader=None, evaluate_every=50, max_step=10):
    train_losses = []
    train_accuracy = []

    network.train()
    device = next(network.parameters()).device

    step = 0

    while True:
        if step > max_step:
            break
        for x, y, cls_mask, pad_mask, segments, _ in tqdm(train_loader, leave=False):

            y_hat, loss, accuracy = network(x.to(device),
                                            y.to(device),
                                            cls_mask.to(device),
                                            pad_mask.to(device),
                                            segments.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_accuracy.append(accuracy.item())

            step += 1

            if step > max_step:
                break

            print(f'train mean loss = {np.mean(train_losses[-1])}')
            print(f'train mean acc = {np.mean(train_accuracy[-1])}')

            if valid_loader:
                if step % evaluate_every == 0:
                    network.eval()
                    val_losses = []
                    val_accuracy = []
                    print('validation...')
                    with torch.no_grad():
                        for x_val, y_val, cls_mask_val, pad_mask_val, segment_val, _ in tqdm(valid_loader, leave=False):
                            y_hat, loss, accuracy = network(x_val.to(device),
                                                            y_val.to(device),
                                                            cls_mask_val.to(device),
                                                            pad_mask_val.to(device),
                                                            segment_val.to(device))
                            val_losses.append(loss.item())
                            val_accuracy.append(accuracy.item())

                    network.train()

                    torch.save(network.state_dict(), f'model_step_{step}.pt')

                    print(f'val mean loss = {np.mean(val_losses)}')
                    print(f'val mean acc = {np.mean(val_accuracy)}')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model", help='Name of the model',
                        default='bert-base-multilingual-uncased')
    parser.add_argument("--device", help='Device name',
                        default='cpu')
    parser.add_argument("--train_path", help='Path of the training data folder',
                        default=r'sample_data\train')
    parser.add_argument("--valid_path", help='Path of the validation data folder',
                        default=r'sample_data\valid')
    parser.add_argument("--max_step", help='Maximum number of step',
                        default=200)
    parser.add_argument("--evaluate_every", help='',
                        default=50)
    parser.add_argument("--lr", help='Learning rate',
                        default=1e-3)
    parser.add_argument("--train_Bs", help='Training batch size',
                        default=16)
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
    ps = get_parser()
    args = ps.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    data = ExtDataset(tokenizer, path=args.train_path)
    valid = ExtDataset(tokenizer, path=args.valid_path)

    loader = DataLoader(data, batch_size=int(args.train_Bs))
    v_loader = DataLoader(valid, batch_size=int(args.val_Bs))

    if args.baseline:
        model = BaselineEXT(num_emb=tokenizer.vocab_size,
                            emb_dim=64,
                            n_head=int(args.n_head),
                            num_layers=int(args.num_layers)).to(device=args.device)
    else:
        model = BertExtSum(bert_model=args.bert_model,
                           add_transformer_layers=args.add_transformer_layers,
                           n_head=int(args.n_head),
                           num_layers=int(args.num_layers)).to(device=args.device)

    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    train(model, opt, loader, max_step=int(args.max_step), valid_loader=v_loader,
          evaluate_every=int(args.evaluate_every))
