import numpy as np
import torch


class DataLoader(object):
    def __init__(self, path="data/"):
        self.path = path
        self.allPos = []
        self.testPos = []
        self.m_item = 0
        self.cnt = 0
        with open(path + "train.txt", "r") as f:
            for line in f.readlines():
                x = list(map(int, line.split(' ')))
                self.allPos.append([])
                for item_id in x[1:]:
                    self.m_item = max(self.m_item, item_id + 1)
                    self.cnt += 1
                    self.allPos[-1].append(item_id)
        self.n_user = len(self.allPos)

        with open(path + "test.txt", "r") as f:
            for line in f.readlines():
                x = list(map(int, line.split(' ')))
                self.testPos.append(x[1])

    def generate_data(self, batch_size):
        users = np.random.randint(0, self.n_user, self.cnt)

        for i in range(0, self.n_user, batch_size):
            pos_items, neg_items = [], []
            for user in users[i: i+batch_size]:
                pos_items.append(np.random.choice(self.allPos[user]))
                t = np.random.randint(self.m_item)
                while t in self.allPos[user]:
                    t = np.random.randint(self.m_item)
                neg_items.append(t)
            yield users[i: i+batch_size], pos_items, neg_items

    def evaluate(self, model, device):
        auc = []
        for user_id in range(self.n_user):
            users = [user_id] * 100
            items = [self.testPos[user_id]]
            for i in range(99):
                t = np.random.randint(self.m_item)
                while t in self.allPos[user_id] or t == items[0]:
                    t = np.random.randint(self.m_item)
                items.append(t)
            users = torch.Tensor(users).long().to(device)
            items = torch.Tensor(items).long().to(device)
            scores = model.forward(users, items).cpu().numpy()
            s = 0
            for score in scores[1:]:
                if scores[0] > score:
                    s += 1
            auc.append(s / 99)
        return np.mean(auc)




