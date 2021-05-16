import torch
from data import DataLoader
from bprmf import BprMF


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    dl = DataLoader()
    model = BprMF(dl.n_user, dl.m_item, 32, 0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    print("Data Size: n_user = %d, m_item = %d, cnt = %d" % (dl.n_user, dl.m_item, dl.cnt))
    print("Device =", device)

    with torch.no_grad():
        model = model.eval()
        auc = dl.evaluate(model, device)
    print("Epoch 0: AUC =", auc)

    for epoch in range(200):
        total_loss = 0
        model = model.train()
        for users, pos_items, neg_items in dl.generate_data(batch_size=256):
            users = torch.Tensor(users).long().to(device)
            pos_items = torch.Tensor(pos_items).long().to(device)
            neg_items = torch.Tensor(neg_items).long().to(device)

            loss = model.bpr_loss(users, pos_items, neg_items)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.cpu().item()
        with torch.no_grad():
            model = model.eval()
            auc = dl.evaluate(model, device)
        print("Epoch %d: loss = %f, AUC = %f" % (epoch + 1, total_loss, auc))


if __name__ == "__main__":
    main()
