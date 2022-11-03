import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    block1 = nn.Sequential(
        nn.Linear(in_features=dim, out_features=hidden_dim),
        norm(dim=hidden_dim),
        nn.ReLU(),
        nn.Dropout(p=drop_prob),
        nn.Linear(in_features=hidden_dim, out_features=dim),
        norm(dim=dim),
    )

    res_block = nn.Residual(block1)
    full_block = nn.Sequential(
        res_block,
        nn.ReLU()
    )

    return full_block


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    out1 = nn.Sequential(
        nn.Linear(in_features=dim, out_features=hidden_dim),
        nn.ReLU(),
    )

    res_blocks = [ResidualBlock(dim=hidden_dim, hidden_dim=hidden_dim//2, norm=norm, drop_prob=drop_prob)
                  for _ in range(num_blocks)]
    res_part = nn.Sequential(*res_blocks)

    net = nn.Sequential(
        out1,
        res_part,
        nn.Linear(in_features=hidden_dim, out_features=num_classes)
    )

    return net


def train_epoch(dataloader, model, opt):
    model.train()

    loss_f = nn.SoftmaxLoss()
    losses = []
    errors = 0

    for batch_x, batch_y in dataloader:
        opt.reset_grad()
        h = model(batch_x)
        loss = loss_f(h, batch_y)
        losses.append(loss.numpy())
        errors += np.sum(np.argmax(h.numpy(), axis=1) != batch_y.numpy())

        loss.backward()
        opt.step()

    average_loss = np.mean(losses)
    error_rate = errors / len(dataloader.dataset)

    return error_rate, average_loss


def eval_epoch(dataloader, model):
    losses = []
    errors = 0
    model.eval()
    loss_f = nn.SoftmaxLoss()

    for batch_x, batch_y in dataloader:
        h = model(batch_x)
        loss = loss_f(h, batch_y)
        losses.append(loss.numpy())
        errors += np.sum(np.argmax(h.numpy(), axis=1) != batch_y.numpy())

    average_loss = np.mean(losses)
    error_rate = errors / len(dataloader.dataset)

    return error_rate, average_loss


def epoch(dataloader, model, opt=None):
    np.random.seed(4)

    if opt is None:
        return eval_epoch(dataloader, model)
    else:
        return train_epoch(dataloader, model, opt)


def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)

    train_x_path = os.path.join(data_dir, "train-images-idx3-ubyte.gz")
    train_y_path = os.path.join(data_dir, "train-labels-idx1-ubyte.gz")

    test_x_path = os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
    test_y_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")

    train_dataset = ndl.data.MNISTDataset(train_x_path, train_y_path)
    test_dataset = ndl.data.MNISTDataset(test_x_path, test_y_path)

    train_dataloader = ndl.data.DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataloader = ndl.data.DataLoader(test_dataset, batch_size, shuffle=False)

    model = MLPResNet(dim=784, hidden_dim=hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    avg_loss_epoch = 0
    error_rate_epoch = 0

    for e in range(epochs):
        print("epoch:", e)
        error_rate_epoch, avg_loss_epoch = epoch(train_dataloader, model, opt)
        print(avg_loss_epoch)

    err_rate_test, avg_loss_test = epoch(test_dataloader, model)
    return error_rate_epoch, avg_loss_epoch, err_rate_test, avg_loss_test,


if __name__ == "__main__":
    train_mnist(data_dir="../data")
