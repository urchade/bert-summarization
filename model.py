import numpy as np
import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy as bce_loss
from transformers import AutoModel

N_MAX_POSITIONS = 512


def create_sinusoidal_embeddings(n_pos, dim, out): # Same as (Vaswani et al., 2017)
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
        for pos in range(n_pos)
    ])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False


class TransformerExt(nn.Module):
    def __init__(self, d_model, n_head, num_layers, **kwargs):
        super().__init__()

        self.position_embeddings = nn.Embedding(N_MAX_POSITIONS, d_model)
        create_sinusoidal_embeddings(N_MAX_POSITIONS, d_model, self.position_embeddings.weight.data)

        self.segment_embeddings = nn.Embedding(2, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, **kwargs)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, segments=None, pad_mask=None):
        batch_size, max_len, d_model = x.size()

        positions = torch.arange(max_len, device=x.device)
        pos_emb = self.position_embeddings(positions).expand(size=(batch_size, max_len, d_model))

        x = x + pos_emb

        if segments is not None:
            seg_emb = self.position_embeddings(segments)
            x = x + seg_emb

        # pd mask have be of shape (seq_len, batch_size)
        if pad_mask is not None:
            pad_mask = pad_mask.transpose(1, 0)

        # we can also add a layerNorm

        out = self.transformer_encoder(x, src_key_padding_mask=pad_mask)

        return out


@torch.no_grad()
def compute_accuracy(outputs, y, mask): # Not really useful because most of the sentences are not summary
    predicted = (outputs > 0.5).float() * 1
    acc = ((predicted == y).float() * mask).sum()
    return acc / mask.sum()


def compute_loss(outputs, y, mask):
    masked_loss = bce_loss(outputs, y.float(), reduction='none') * mask.float()  # (batch_size, max_len)
    return masked_loss.sum() / mask.sum()


class BertExtSum(nn.Module):
    def __init__(self, bert_model, add_transformer_layers, n_head,
                 num_layers):
        super().__init__()

        self.add_transformer_layers = add_transformer_layers
        self.bert = AutoModel.from_pretrained(bert_model)
        self.hidden_size = self.bert.config.hidden_size

        if self.add_transformer_layers:
            self.intermediate_layer = TransformerExt(self.hidden_size, n_head=n_head, num_layers=num_layers)

        self.classifier = nn.Sequential(nn.Linear(self.hidden_size, 1),
                                        nn.Sigmoid())

    def forward(self, x, y, mask_cls, pad_mask=None, segments=None):

        # The first element corresponds to the final representation of all tokens
        x = self.bert(x, attention_mask=pad_mask, token_type_ids=segments)[0]  # (batch_size, max_len, hidden_size)

        if self.add_transformer_layers:
            x = self.intermediate_layer(x, pad_mask=pad_mask)  # (batch_size, max_len, hidden_size)

        x = self.classifier(x)  # (batch_size, max_len, 1)
        y_hat = x.squeeze(-1)  # (batch_size, max_len)

        # For every token, we have a probability but we are only interested by [CLS] tokens
        # compute the loss
        loss = compute_loss(y_hat, y, mask_cls)
        accuracy = compute_accuracy(y_hat, y, mask_cls)

        return y_hat, loss, accuracy


class BaselineEXT(nn.Module):
    def __init__(self, num_emb, emb_dim, n_head, num_layers):
        super().__init__()

        self.embedding = nn.Embedding(num_emb, emb_dim)

        self.transformer = TransformerExt(emb_dim, n_head, num_layers)

        self.fc = nn.Sequential(nn.Linear(emb_dim, 1),
                                nn.Sigmoid())

    def forward(self, x, y, mask_cls, pad_mask=None, segments=None):

        x = self.embedding(x)
        x = self.transformer(x, pad_mask=pad_mask, segments=segments)
        y_hat = self.fc(x).squeeze(-1)
        loss = compute_loss(y_hat, y, mask_cls)
        accuracy = compute_accuracy(y_hat, y, mask_cls)

        return y_hat, loss, accuracy

