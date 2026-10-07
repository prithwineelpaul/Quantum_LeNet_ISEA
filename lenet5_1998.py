"""
LeNet-5 -- faithful PyTorch reimplementation of
Y. LeCun, L. Bottou, Y. Bengio, P. Haffner, "Gradient-Based Learning Applied to
Document Recognition," Proc. IEEE 86(11):2278-2324, 1998.  doi:10.1109/5.726791

Architecture (Sec. II-B of the paper):
  Input : 32x32 (MNIST 28x28 padded by 2). Background = -0.1, foreground = 1.175
  C1    : 6 maps 28x28, 5x5 conv                                   156 params
  S2    : 6 maps 14x14, 2x2 sum * trainable coef + trainable bias, squash   12
  C3    : 16 maps 10x10, 5x5 conv, sparse connection table (Table I)       1,516
  S4    : 16 maps 5x5, trainable subsampling                               32
  C5    : 120 maps 1x1, 5x5 conv (full connection to S4)                48,120
  F6    : 84 units, fully connected                                    10,164
  Output: 10 Euclidean RBF units, y_i = sum_j (x_j - w_ij)^2, with FIXED w_i
          (7x12 = 84-pixel +/-1 bitmaps of the digits)
  Squash: f(a) = 1.7159 * tanh(2/3 * a)                       total = 60,000
  Loss  : MAP criterion  E = y_D + log(exp(-j) + sum_i exp(-y_i))   (Eq. 9)

Deviations (stated, not hidden):
  * The paper's exact 7x12 RBF bitmaps are only shown as a figure; here they are
    rendered from a bitmap font and binarised to +/-1.
  * The paper trains with stochastic diagonal Levenberg-Marquardt; plain SGD is
    used here, following the paper's 20-pass step schedule shape.
Reported results: 0.95% test error (no distortions), 0.8% with distortions.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

A, S = 1.7159, 2.0 / 3.0


def squash(x):
    return A * torch.tanh(S * x)


def uniform_init(shape, fan_in):
    # paper (Appendix A): uniform in [-2.4/F_in, 2.4/F_in]
    return nn.Parameter((torch.rand(*shape) * 2 - 1) * 2.4 / fan_in)


# Table I: which S2 maps feed each C3 map
C3_TABLE = [
    [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 0], [5, 0, 1],
    [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 0], [4, 5, 0, 1], [5, 0, 1, 2],
    [0, 1, 3, 4], [1, 2, 4, 5], [0, 2, 3, 5],
    [0, 1, 2, 3, 4, 5],
]


class Subsample(nn.Module):
    """S2 / S4: sum of 2x2 field * trainable coefficient + trainable bias, squashed."""

    def __init__(self, maps):
        super().__init__()
        self.coef = nn.Parameter(torch.full((maps,), 0.25))   # starts as a 2x2 average
        self.bias = nn.Parameter(torch.zeros(maps))

    def forward(self, x):
        s = F.avg_pool2d(x, 2) * 4.0                          # 2x2 sum
        return squash(s * self.coef.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1))


class C3(nn.Module):
    """Sparsely connected conv layer: one 5x5 kernel set per output map."""

    def __init__(self):
        super().__init__()
        self.w = nn.ParameterList(
            [uniform_init((1, len(src), 5, 5), 25 * len(src)) for src in C3_TABLE])
        self.b = nn.Parameter(torch.zeros(16))
        for j, src in enumerate(C3_TABLE):
            self.register_buffer(f"src{j}", torch.tensor(src))

    def forward(self, x):
        outs = [F.conv2d(x[:, getattr(self, f"src{j}")], self.w[j]) for j in range(16)]
        return torch.cat(outs, 1) + self.b.view(1, -1, 1, 1)


def rbf_bitmaps():
    """10 x 84 fixed RBF centres: digits rendered as 7x12 bitmaps, values +/-1."""
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.load_default()
    codes = []
    for d in range(10):
        img = Image.new("L", (28, 48), 0)
        ImageDraw.Draw(img).text((8, 14), str(d), fill=255, font=font)
        arr = np.array(img)
        ys, xs = np.nonzero(arr)
        crop = img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
        bm = np.array(crop.resize((7, 12), Image.NEAREST)) > 127
        codes.append(np.where(bm, 1.0, -1.0).reshape(-1))
    return torch.tensor(np.stack(codes), dtype=torch.float32)


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.C1w = uniform_init((6, 1, 5, 5), 25)
        self.C1b = nn.Parameter(torch.zeros(6))
        self.S2 = Subsample(6)
        self.C3 = C3()
        self.S4 = Subsample(16)
        self.C5w = uniform_init((120, 16, 5, 5), 16 * 25)
        self.C5b = nn.Parameter(torch.zeros(120))
        self.F6 = nn.Linear(120, 84)
        with torch.no_grad():
            self.F6.weight.copy_(uniform_init((84, 120), 120))
            self.F6.bias.zero_()
        self.register_buffer("rbf", rbf_bitmaps())           # fixed, not trained

    def forward(self, x):
        x = squash(F.conv2d(x, self.C1w, self.C1b))          # 6x28x28
        x = self.S2(x)                                       # 6x14x14
        x = squash(self.C3(x))                               # 16x10x10
        x = self.S4(x)                                       # 16x5x5
        x = squash(F.conv2d(x, self.C5w, self.C5b))          # 120x1x1
        x = squash(self.F6(x.flatten(1)))                    # 84
        # RBF output: squared distance to each class bitmap (penalty, lower = better)
        return ((x.unsqueeze(1) - self.rbf.unsqueeze(0)) ** 2).sum(-1)   # N x 10


def map_loss(y, target, j=0.1):
    """Eq. (9): E = y_D + log(exp(-j) + sum_i exp(-y_i)), averaged over the batch."""
    y_d = y.gather(1, target.view(-1, 1)).squeeze(1)
    lse = torch.logsumexp(torch.cat([torch.full_like(y[:, :1], -j), -y], 1), 1)
    return (y_d + lse).mean()


def get_loaders(root, batch_size):
    # [0,1] -> background -0.1, foreground 1.175; pad 28x28 -> 32x32 with background
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t * 1.275 - 0.1),
        transforms.Lambda(lambda t: F.pad(t, (2, 2, 2, 2), value=-0.1)),
    ])
    train = datasets.MNIST(root, train=True, download=True, transform=tf)
    test = datasets.MNIST(root, train=False, download=True, transform=tf)
    return (DataLoader(train, batch_size=batch_size, shuffle=True),
            DataLoader(test, batch_size=1000, shuffle=False))


def lr_at(epoch, base):
    # paper's schedule (0.0005 x2, 0.0002 x3, 0.0001 x3, 0.00005 x4, 0.00001 rest),
    # expressed relative to `base`
    rel = [1.0] * 2 + [0.4] * 3 + [0.2] * 3 + [0.1] * 4
    return base * (rel[epoch] if epoch < len(rel) else 0.02)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    err, n, loss = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss += map_loss(out, y).item() * y.size(0)
        err += (out.argmin(1) != y).sum().item()           # smallest penalty wins
        n += y.size(0)
    return loss / n, 100.0 * err / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="./data")
    p.add_argument("--epochs", type=int, default=20)   # paper: 20 passes
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--batch", type=int, default=1)     # paper: per-sample updates
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LeNet5().to(device)
    print(f"trainable parameters: {sum(q.numel() for q in model.parameters())}  (paper: 60,000)")

    train_loader, test_loader = get_loaders(args.data, args.batch)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        for g in opt.param_groups:
            g["lr"] = lr_at(epoch, args.lr)
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = map_loss(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        tr = evaluate(model, train_loader if args.batch >= 64 else
                      DataLoader(train_loader.dataset, batch_size=1000), device)
        te = evaluate(model, test_loader, device)
        print(f"pass {epoch + 1:2d} | lr {lr_at(epoch, args.lr):.5f} | "
              f"train loss {tr[0]:.4f} err {tr[1]:5.2f}% | test loss {te[0]:.4f} err {te[1]:5.2f}%")

    torch.save(model.state_dict(), "lenet5_1998.pt")


if __name__ == "__main__":
    main()
