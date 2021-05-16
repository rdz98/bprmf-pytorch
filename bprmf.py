import torch
import torch.nn as nn


class BprMF(nn.Module):
    def __init__(self, n_user, m_item, dim, reg):
        super(BprMF, self).__init__()
        self.n_user = n_user
        self.m_item = m_item
        self.dim = dim
        self.reg = reg

        self.Embedding_User = nn.Embedding(n_user, dim)
        self.Embedding_Item = nn.Embedding(m_item, dim)
        nn.init.normal_(self.Embedding_User.weight, std=0.01)
        nn.init.normal_(self.Embedding_Item.weight, std=0.01)

    def bpr_loss(self, users, pos_items, neg_items):
        users_emb = self.Embedding_User(users)
        pos_items_emb = self.Embedding_Item(pos_items)
        neg_items_emb = self.Embedding_Item(neg_items)

        pos_scores = torch.sum(users_emb * pos_items_emb, dim=-1)
        neg_scores = torch.sum(users_emb * neg_items_emb, dim=-1)
        loss = (pos_scores - neg_scores).sigmoid().log().mean()
        return -loss

    def forward(self, users, items):
        users_emb = self.Embedding_User(users)
        items_emb = self.Embedding_Item(items)
        scores = torch.sum(users_emb * items_emb, dim=-1)
        return scores
