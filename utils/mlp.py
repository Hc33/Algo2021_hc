from numpy.core.fromnumeric import shape
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.dataloader import Dataset, DataLoader


class NN_works(object):
    def __init__(self, model, train_feature, train_label, test_feature, loss_func, optimizer, epoch):
        self.model = model.cuda()
        self.train_feature = train_feature
        self.train_label = train_label
        self.test_feature = test_feature
        self.output_size = (test_feature.shape[0], train_label.shape[1])
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.epoch = epoch
        assert train_feature.shape[0] == train_label.shape[0]
        assert train_feature.shape[1] == test_feature.shape[1]

    def fit(self):  # training
        ds_train = TrDs(self.train_feature, self.train_label)
        dl_train = DataLoader(ds_train, 32, True, num_workers=4)
        _train(self.model, self.loss_func, self.optimizer, self.epoch, dl_train)

    def predict(self): # predict
        ds_val = ValDs(self.test_feature)
        dl_val = DataLoader(ds_val, 16, False, num_workers=4)
        output = _test(self.model, dl_val, output_size=self.output_size)
        return output



class Mlp(nn.Module):
    def __init__(self, in_, out_):
        super(Mlp, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(inplace=True),

            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ELU(inplace=True),

            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ELU(inplace=True),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(inplace=True),

            nn.Linear(1024, out_),
        )
        
    def forward(self, x):
        out = self.model(x)
        return out


def _train(model, loss_func, optimizer, epoch, dl):
    model.train()
    for e in range(epoch):
        epoch_loss = 0
        step = 0
        for d in dl:
            step += 1
            x = d[0].to(torch.float32).cuda()
            y = d[1].to(torch.float32).cuda()
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_func(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print("Epoch {0:2} Loss: {1:.4f}".format(e, epoch_loss/step))


def _test(model, dl, output_size):
    model.eval()
    output = np.zeros(output_size,dtype=np.float32)
    # only once
    with torch.no_grad():
        for i,b in enumerate(dl):
            x = b.to(torch.float32).cuda()
            out = model(x).cpu().numpy().squeeze() # cpu
            output[i*dl.batch_size : i*dl.batch_size+out.shape[0]] = out
    return output

class TrDs(Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
        assert data_x.shape[0] == data_y.shape[0]
    
    def __len__(self):
        return self.data_x.shape[0]

    def __getitem__(self, i):
        x = self.data_x[i]
        y = self.data_y[i]
        return x, y


class ValDs(Dataset):
    def __init__(self, data_x):
        self.data_x = data_x

    def __len__(self):
        return self.data_x.shape[0]

    def __getitem__(self, i):
        return self.data_x[i]



if __name__ == "__main__":
    import numpy as np
    x = np.random.randn(900, 50) # (n_samples, n_features)

    y = np.random.randn(900, 20) # label
    y_ = np.random.randn(900, 20) # label 2 
    x_val = np.random.randn(100, 50)

    from ols import vectorized_correlation
    print("Score calculating...")
    print("Score is {:.4f}".format(vectorized_correlation(y, y_).mean()))

    model = Mlp(x.shape[1], y.shape[1]).cuda() # 50 -> 20

    ds_train = TrDs(x, y)
    ds_val = ValDs(x)
    dl_train = DataLoader(ds_train, 32, True, num_workers=4)
    dl_val = DataLoader(ds_val, 16, False, num_workers=2)
    
    loss_func = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), 0.001)
    model = _train(model, loss_func, optimizer, 5, dl_train)
    print('Train_finished.')
    output = _test(model, dl_val, output_size=(900, 20))
    print("Score after Training...")
    print("Score is {:.4f}".format(vectorized_correlation(y, output).mean()))
    print('ok')
