import os
import sys
import time
import numpy as np
from tqdm import tqdm
from torch import optim
import torch, torch.nn as nn, math
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset


# ================== Fix the random seed ==================
save_path = r"F:\data_viscid\2D"
checkpoint_file = os.path.join(save_path, "checkpoint_FERAMLP")

sparse_index_file = os.path.join(
    save_path, "sparse_indices_train_10pct.pt"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== Fix the random seed ==================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(4)

# ================== Model definition ==================
class GaussianFourierEncoder(nn.Module):

    def __init__(self, in_dim=2, scales=(1.0, 3.0), ks=(8, 8)):
        super().__init__()
        assert len(scales) == len(ks)

        self.B_list = nn.ParameterList()
        out_dim = 0

        for sigma, K in zip(scales, ks):
            # Initialize in the same way as the original implementation: randn * sigma
            B = torch.randn(K, in_dim) * float(sigma)
            self.B_list.append(nn.Parameter(B))
            out_dim += 2 * K

        self.out_dim = out_dim

    def forward(self, xy):
        outs = []
        for B in self.B_list:
            proj = 2 * math.pi * (xy @ B.T)               # (N,K)
            outs.append(torch.sin(proj))
            outs.append(torch.cos(proj))
        return torch.cat(outs, dim=-1)

class SEBlock(nn.Module):
    def __init__(self, hidden_dim, reduction=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // reduction),
            nn.SiLU(),
            nn.Linear(hidden_dim // reduction, hidden_dim),
            nn.Sigmoid())

    def forward(self, x):
        b, c = x.shape
        w = self.fc(self.gap(x.unsqueeze(-1)).view(b, c))
        return x * w

class RAMLP_WithFourier_NoGate(nn.Module):

    def __init__(self, hidden=256, num_layers=4, out_dim=3,
                 fourier_scales=(1.0, 3.0), ks=(8, 8)):
        super().__init__()

        self.enc = GaussianFourierEncoder(2, fourier_scales, ks)


        mlp_in = 7 + self.enc.out_dim

        self.input_layer = nn.Linear(mlp_in, hidden)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(num_layers - 1)])
        self.se_layers = nn.ModuleList(
            [SEBlock(hidden) for _ in range(num_layers - 1)])
        self.output_layer = nn.Linear(hidden, out_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        latent = x[:, 0:4]
        Ma = x[:, 4:5]
        AoA = x[:, 5:6]
        xy = x[:, 6:8]
        d = x[:, 8:9]

        lowfreq = torch.cat([latent, Ma, AoA, d], dim=-1)  # (N,7)

        fourier = self.enc(xy)                             # (N,F)

        feats = torch.cat([lowfreq, fourier], dim=-1)

        h = self.act(self.input_layer(feats))
        for lin, se in zip(self.hidden_layers, self.se_layers):
            residual = h
            h = self.act(lin(h))
            h = se(h)
            h = h + residual

        return self.output_layer(h)


# ================== Data Loading ==================
path_input = r"F:\data_viscid\2D\input.pt"
path_output = r"F:\data_viscid\2D\label.pt"

input = torch.load(path_input).float()
output = torch.load(path_output).float()

print(input.shape)
print(output.shape)

# ================== Training/validation split ==================
train_size = int(0.8 * input.shape[0])
val_size = input.shape[0] - train_size

dataset = TensorDataset(input, output)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# ================== Sparse training ==================
sparse_ratio = 0.10
num_sparse = max(1, int(len(train_dataset) * sparse_ratio))
print(f"[Sparse training] {num_sparse}/{len(train_dataset)} samples "
      f"({100*sparse_ratio:.1f}%)")

# Sampled only once before training (fixed subset)
if os.path.exists(sparse_index_file):
    print(f"[Sparse training] Loading indices from {sparse_index_file}")
    indices = torch.load(sparse_index_file)
else:
    print(f"[Sparse training] Creating new sparse indices ({100*sparse_ratio:.1f}%)")
    indices = torch.randperm(len(train_dataset))[:num_sparse]
    torch.save(indices, sparse_index_file)
    print(f"[Sparse training] Saved indices to {sparse_index_file}")
sparse_train_dataset = Subset(train_dataset, indices)

# ================== DataLoader ==================
batch_size = 50

train_loader = DataLoader(
    sparse_train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

# ================== Training settings ==================
num_epochs = 500
learning_rate = 5e-4

model_MLP = RAMLP_WithFourier_NoGate().to(device)
criterion = nn.MSELoss(reduction="mean")
optimizer = optim.Adam(model_MLP.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

tra_losses = []
val_losses = []
train_time_per_epoch, test_time_per_epoch = [], []

def load_train_state(filename, model, optimizer, scheduler):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    loss_record = checkpoint["loss_record"]
    return checkpoint["epoch"], model, optimizer, scheduler, loss_record

def save_train_state(filename, model, optimizer, scheduler, epoch, loss_record):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "loss_record": loss_record}, filename)

if os.path.exists(checkpoint_file):
    start_epoch, model_MLP, optimizer, scheduler, tra_losses = \
        load_train_state(checkpoint_file, model_MLP, optimizer, scheduler)
else:
    start_epoch = 0

# ================== Training loop ==================
for epoch in range(start_epoch, num_epochs):
    model_MLP.train()
    loss1 = 0
    num_batch = 0
    epoch_start_time = time.time()

    for data_train_input, data_train_output in tqdm(train_loader, file=sys.stdout):
        data_train_input = data_train_input.to(device)
        data_train_output = data_train_output.to(device)

        outputs = model_MLP(data_train_input)
        loss = criterion(outputs, data_train_output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss1 += loss.item()
        num_batch += 1

    scheduler.step()
    train_time = time.time() - epoch_start_time
    train_time_per_epoch.append(train_time)

    loss1 = loss1 / max(num_batch, 1)
    tra_losses.append(loss1)

    save_train_state(checkpoint_file, model_MLP, optimizer, scheduler, epoch + 1, tra_losses)

    # ---------- 验证 ----------
    model_MLP.eval()
    loss2 = 0
    num_batch = 0
    epoch_start_time = time.time()

    with torch.no_grad():
        for data_input, data_output in tqdm(val_loader, file=sys.stdout):
            data_input = data_input.to(device)
            data_output = data_output.to(device)

            outputs = model_MLP(data_input)
            loss = criterion(outputs, data_output)

            loss2 += loss.item()
            num_batch += 1

    test_time = time.time() - epoch_start_time
    test_time_per_epoch.append(test_time)

    loss2 = loss2 / max(num_batch, 1)
    val_losses.append(loss2)

    tqdm.write(
        f"Epoch [{epoch+1}/{num_epochs}], "
        f"train_Loss: {loss1:.10f}, train_time: {train_time:.2f}s, "
        f"val_Loss: {loss2:.10f}, val_time: {test_time:.2f}s")

    torch.save(model_MLP, os.path.join(save_path, "model_FERAMLP.pth"))
    torch.save(tra_losses, os.path.join(save_path, "tra_losses_FERAMLP.pth"))
    torch.save(val_losses, os.path.join(save_path, "val_losses_FERAMLP.pth"))

def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = seconds % 60
    return f"{minutes}m {seconds:.2f}s"

tqdm.write(
    f'Total train time: {format_time(sum(train_time_per_epoch))}, '
    f'Total validation time: {format_time(sum(test_time_per_epoch))}')
