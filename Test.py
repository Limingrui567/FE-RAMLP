# -*- coding: utf-8 -*-
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
import torch, torch.nn as nn, math
from scipy.interpolate import griddata


# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the plotting font
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'


# Paths and global parameters
path_data_fluent = r"F:\data\2D\fluent.txt"
path_coords_d    = r"F:\data\2D\mesh_coord_naca4421.dat"
path_coords      = r"F:\data\2D\naca4421.dat"
path_model       = r"F:\data\2D\model_FERAMLP.pth"
path_latent      = r"F:\data\2D\latent.pt"
num_wing    = 3
# Airfoil index mapping
# 0: NACA0006
# 1: NACA1421
# 2: NACA2410
# 3: NACA4421
# 4: NACA6410
# 5: RAE5213
# 6: S4083
Ma_val, AoA_val = 0.4, 4

# Regular grid
nx, ny = 600, 600
xg = np.linspace(-1.2, 0.1, nx)
yg = np.linspace(-0.3, 0.3, ny)
xi, yi = np.meshgrid(xg, yg)

# Colorbar range
levels_u = np.linspace(-0.7, 0.1, 88)
ticks_u  = np.linspace(-0.7, 0.1, 9)

# Input/output normalization
input_min_np = np.array([0.2, -5.0, -2.5, -1.5], dtype=np.float32)  # (Ma, AoA, x, y)
input_max_np = np.array([0.6,  5.0,  1.5,  1.5], dtype=np.float32)
output_min   = np.array([-6, -100., -400], dtype=np.float64)
output_max   = np.array([2, 500.,  400], dtype=np.float64)

def normalize_to_minus1_1(X, min_vals, max_vals):
    return 2 * (X - min_vals) / (max_vals - min_vals) - 1

def np_norm_m11(X, vmin, vmax):
    return 2.0*(X - vmin)/(vmax - vmin) - 1.0

# Load airfoil coordinates
def read_airfoil_dat(path):
    data = []
    with open(path, "r") as f:
        lines = f.readlines()[2:]
    for line in lines:
        if line.strip() == "":
            continue
        nums = line.strip().split()
        row = [float(x) for x in nums[:-1]]
        data.append(row)
    coords = np.asarray(data, dtype=np.float64)
    return coords

def read_airfoil_dat_d(path):
    coords = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(1, 2))
    return torch.tensor(coords, dtype=torch.float32)

coords_coords = read_airfoil_dat(path_coords)
coords_coords_d = read_airfoil_dat_d(path_coords_d)
curve_points  = np.column_stack((coords_coords[:, 0] - 1.0, coords_coords[:, 1]))  # x-1

if np.any(curve_points[0] != curve_points[-1]):
    curve_closed = np.vstack([curve_points, curve_points[0]])
else:
    curve_closed = curve_points.copy()
poly_path = Path(curve_points)

# Load Fluent data and normalize to [-1, 1]
raw = np.loadtxt(path_data_fluent, delimiter=",", skiprows=1)[:, 1:]
print(raw)
fluent_x = raw[:, 0:1].astype(np.float64)
fluent_y = raw[:, 1:2].astype(np.float64)
output   = raw[:, 2:].astype(np.float64)
output = np_norm_m11(output, output_min, output_max)
fluent_u = output[:, 1:2] # According to fluent.txt: 0 → Cp, 1 → U, 2 → V
print(np.max(output))
print(np.min(output))

# First process Fluent scattered points
xy_all = np.column_stack([fluent_x.ravel(), fluent_y.ravel()])
inside = poly_path.contains_points(xy_all, radius=+1e-6)
outside_only = ~inside
pts_used  = xy_all[outside_only]
uF_used   = fluent_u.ravel()[outside_only]

# Fluent: interpolate the same scattered points onto the grid (linear)
zi_fluent = griddata((pts_used[:, 0], pts_used[:, 1]), uF_used, (xi, yi), method='linear')

# Set NaN inside the grid
for i in range(xi.shape[0]):
    for j in range(xi.shape[1]):
        if poly_path.contains_point((xi[i, j], yi[i, j]), radius=-1e-6):
            zi_fluent[i, j] = np.nan

# Plot Fluent contour
plt.figure(figsize=(9, 4))
plt.rcParams['font.family'] = 'Times New Roman'
contour = plt.contourf(xi, yi, zi_fluent, cmap="bwr", levels=levels_u)
cbar = plt.colorbar(); cbar.set_ticks(ticks_u)
xticks = np.linspace(-1.2, 0.1, num=5)
plt.xticks(xticks, labels=[f"{x+1:.2f}" for x in xticks])
plt.yticks(np.linspace(-0.3, 0.3, num=6))
plt.xlabel('x', fontstyle='italic', fontsize=15)
plt.ylabel('y', fontstyle='italic', fontsize=15)
plt.title('CFD(U-Velocity)', fontfamily='Times New Roman', fontsize=15, fontweight='bold', fontstyle='italic')
plt.savefig(r"F:\data\2D\cfd-u.tif", dpi=300, format="tiff",
            bbox_inches="tight", pad_inches=0.1)
plt.show()

# Define the random seed
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(4)

# Model definition
class GaussianFourierEncoder(nn.Module):
    def __init__(self, in_dim=2, scales=(1.0, 3.0), ks=(8, 8), seed=42):
        super().__init__()
        assert len(scales) == len(ks)
        torch.manual_seed(seed)

        self.B_list = nn.ParameterList()
        out_dim = 0

        for sigma, K in zip(scales, ks):
            # 用与你原版一致的方式初始化：randn * sigma
            B = torch.randn(K, in_dim) * float(sigma)     # (K,2)
            self.B_list.append(nn.Parameter(B))           # 现在可学习
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
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.shape
        w = self.fc(self.gap(x.unsqueeze(-1)).view(b, c))
        return x * w

class RAMLP_WithFourier_NoGate(nn.Module):
    def __init__(self, hidden=256, num_layers=4, out_dim=3,
                 fourier_scales=(1.0, 3.0), ks=(8, 8),
                 w_hidden=64):
        super().__init__()

        self.enc = GaussianFourierEncoder(2, fourier_scales, ks)


        mlp_in = 7 + self.enc.out_dim

        self.input_layer = nn.Linear(mlp_in, hidden)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(num_layers - 1)]
        )
        self.se_layers = nn.ModuleList(
            [SEBlock(hidden) for _ in range(num_layers - 1)]
        )
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

# Define the computation of the unsigned distance
def add_unsigned_min_dist_feature(airfoil_xy: np.ndarray,
                                     inputs: np.ndarray,
                                     xy_cols=(0, 1),
                                     chunk_size: int = 50_000) -> np.ndarray:

    assert airfoil_xy.ndim == 2 and airfoil_xy.shape[1] == 2
    assert inputs.ndim == 2 and len(xy_cols) == 2

    A = np.asarray(airfoil_xy, dtype=np.float64)        # (M,2)
    P = np.asarray(inputs[:, list(xy_cols)], dtype=np.float64)  # (N,2)

    N = P.shape[0]
    d_min = np.empty((N, 1), dtype=np.float64)

    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        Q = P[s:e]                                      # (n,2)
        diff = Q[:, None, :] - A[None, :, :]            # (n,M,2)
        dist2 = np.sum(diff * diff, axis=2)             # (n,M)
        d_min[s:e, 0] = np.sqrt(np.min(dist2, axis=1))  # (n,)

    return np.concatenate([inputs, d_min], axis=1)

xy_all = add_unsigned_min_dist_feature(coords_coords_d, xy_all)
dis_xy = xy_all[:, -1:] / 1.5
xy_all = xy_all[:, 0:2]
model = torch.load(path_model).to(torch.float32).to(device); model.eval()
latent_train = torch.load(path_latent)
latent_max = torch.tensor([10, 8, 6, 15], device=latent_train.device, dtype=latent_train.dtype)
latent_min = torch.tensor([-2, -9, -10, -4], device=latent_train.device, dtype=latent_train.dtype)
latent_train = 2*(latent_train - latent_min)/(latent_max - latent_min) - 1

N_all = xy_all.shape[0]
ma_aoa = np.column_stack([np.full((N_all,1), Ma_val, dtype=np.float32),
                          np.full((N_all,1), AoA_val, dtype=np.float32)])
ma_aoa_norm = np_norm_m11(ma_aoa, input_min_np[:2], input_max_np[:2])
xy_norm_all = np_norm_m11(xy_all.astype(np.float32), input_min_np[2:], input_max_np[2:])
latent_row  = latent_train[num_wing].detach().cpu().numpy().astype(np.float32)[None, :]
latent_rep  = np.repeat(latent_row, N_all, axis=0)
inp_all     = np.concatenate([latent_rep, ma_aoa_norm, xy_norm_all, dis_xy], axis=1)
print(inp_all.shape)

with torch.no_grad():
    u_mlp_all = model(torch.from_numpy(inp_all).float().to(device))[:, 0].cpu().numpy()  # (N_all,)

print(max(u_mlp_all))
print(min(u_mlp_all))
u_fluent_all = fluent_u.ravel()                           # (N_all,)
abs_err_all  = np.abs(u_fluent_all - u_mlp_all)
print("[No-Interp @all pts] mean=%.6f p95=%.6f max=%.6f"
      % (abs_err_all.mean(), np.percentile(abs_err_all,95), abs_err_all.max()))

# interpolation using the same scattered points
uM_used  = u_mlp_all[outside_only]       # (N_used,)
err_used = np.abs(uF_used - uM_used)        # (N_used,)

zi_mlp = griddata((pts_used[:, 0], pts_used[:, 1]), uM_used,  (xi, yi), method='linear')
zi_err = griddata((pts_used[:, 0], pts_used[:, 1]), err_used, (xi, yi), method='linear')

print(np.max(zi_err))
print(np.min(zi_err))

for i in range(xi.shape[0]):
    for j in range(xi.shape[1]):
        if poly_path.contains_point((xi[i, j], yi[i, j]), radius=-1e-6):
            zi_mlp[i, j] = np.nan
            zi_err[i, j] = np.nan

# Plot comparison contour maps
Z_fluent = np.ma.masked_invalid(zi_fluent)
Z_mlp = np.ma.masked_invalid(zi_mlp)
fluent_min = np.nanmin(zi_mlp)
fluent_max = np.nanmax(zi_mlp)
levels = np.linspace(fluent_min, fluent_max, 44)

plt.figure(figsize=(9,4))
plt.rcParams['font.family'] = 'Times New Roman'

CS1 = plt.contour(xi, yi, Z_fluent,
                 levels=levels,
                 colors='blue',
                 linewidths=1,
                 linestyles='--')

CS2 = plt.contour(xi, yi, Z_mlp,
                 levels=levels,
                 colors='red',
                 linewidths=1,
                 linestyles='--')

plt.xlabel('x', fontstyle='italic', fontsize=15)
plt.ylabel('y', fontstyle='italic', fontsize=15)
plt.title('U-Velocity Contour Lines', fontstyle='italic')
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig(r"F:\data\2D\contour_maps_u.tif", dpi=300, format="tiff",
            bbox_inches="tight", pad_inches=0.1)
plt.show()

mask = np.ma.getmaskarray(Z_fluent) | np.ma.getmaskarray(Z_mlp)
T = np.asarray(Z_fluent)[~mask].ravel()
P = np.asarray(Z_mlp)[~mask].ravel()

# Downsampling
max_points = 3000
if T.size > max_points:
    idx = np.linspace(0, T.size - 1, max_points, dtype=int)
    T_plot, P_plot = T[idx], P[idx]
else:
    T_plot, P_plot = T, P

# Pearson R
Tb, Pb = T.mean(), P.mean()
num = np.sum((T - Tb) * (P - Pb))
den = np.sqrt(np.sum((T - Tb)**2) * np.sum((P - Pb)**2))
R = num / den if den > 0 else np.nan

# Start plotting the test results of FE-RAMLP
plt.figure(figsize=(4.2, 4.2))
plt.rcParams['font.family'] = 'Times New Roman'

x_min, x_max = T_plot.min(), T_plot.max()
y_min, y_max = P_plot.min(), P_plot.max()
pad_x = 0.01 * (x_max - x_min + 1e-12)
pad_y = 0.01 * (y_max - y_min + 1e-12)
plt.xlim(x_min - pad_x, x_max + pad_x)
plt.ylim(y_min - pad_y, y_max + pad_y)

# Red markers with black edges
plt.scatter(T_plot, P_plot, s=28, color='red', edgecolors='k',
            linewidths=0.35, alpha=0.95, zorder=3)

# Axis limits
plt.gca().set_aspect('equal', adjustable='box')
x0, x1 = plt.xlim(); y0, y1 = plt.ylim()

# Grid lines: dash-dot style
from matplotlib.ticker import AutoMinorLocator
ax = plt.gca()
ax.grid(True, which='major', linestyle='-.', linewidth=0.8, alpha=0.7)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.grid(True, which='minor', linestyle='-.', linewidth=0.5, alpha=0.35)

#  R label: placed on the diagonal line with a larger normal offset
x0, x1 = plt.xlim(); y0, y1 = plt.ylim()
s_pos = 0.60
rng   = max(x1-x0, y1-y0)

# Offset ratio: the larger the value, the farther from the diagonal (0.12–0.20 recommended)
offset_frac = 0.16
off = (offset_frac * rng) / np.sqrt(2)

# First place it on the diagonal, then shift along the normal direction (-1, +1)
xR = (x0 + s_pos*(x1-x0)) - off
yR = (y0 + s_pos*(y1-y0)) + off

# Boundary protection: prevent the text from going out of bounds
margin = 0.04 * rng
xR = np.clip(xR, x0 + margin, x1 - margin)
yR = np.clip(yR, y0 + margin, y1 - margin)

plt.xlabel('CFD', fontstyle='italic', fontsize=12)
plt.ylabel('FE-RAMLP', fontstyle='italic', fontsize=12)

label = rf'$\mathit{{R}} = {R:.5f}$'
plt.text(xR, yR, label,
         fontsize=12, ha='center', va='center',
         bbox=dict(boxstyle='square,pad=0.35', facecolor='white', edgecolor='k', alpha=0.95))

plt.savefig(r"F:\data_\2D\Correlation_u.tif", dpi=300, format="tiff",
            bbox_inches="tight", pad_inches=0.1)
plt.show()


# Plot FE-RAMLP contours
plt.figure(figsize=(9, 4))
plt.rcParams['font.family'] = 'Times New Roman'
contour = plt.contourf(xi, yi, zi_mlp, cmap="bwr", levels=levels_u)
cbar = plt.colorbar(); cbar.set_ticks(ticks_u)
xticks = np.linspace(-1.2, 0.1, num=5)
plt.xticks(xticks, labels=[f"{x+1:.2f}" for x in xticks])
plt.yticks(np.linspace(-0.3, 0.3, num=6))
plt.xlabel('x', fontstyle='italic', fontsize=15)
plt.ylabel('y', fontstyle='italic', fontsize=15)
plt.title('FE-RAMLP(U-Velocity)', fontfamily='Times New Roman', fontsize=15, fontweight='bold', fontstyle='italic')
plt.savefig(r"F:\data\2D\u.tif", dpi=300, format="tiff",
            bbox_inches="tight", pad_inches=0.1)
plt.show()

# Plot absolute error contours
plt.figure(figsize=(9, 4))
plt.rcParams['font.family'] = 'Times New Roman'
err_max = np.nanpercentile(zi_err, 99) if np.isfinite(zi_err).any() else 0.02
levels_e = np.linspace(0.0, 0.04, 100)
contour = plt.contourf(xi, yi, zi_err, cmap="bwr", levels=levels_e)
cbar = plt.colorbar(); cbar.set_ticks(np.linspace(0.0, 0.04, 5))
xticks = np.linspace(-1.2, 0.1, num=5)
plt.xticks(xticks, labels=[f"{x+1:.2f}" for x in xticks])
plt.yticks(np.linspace(-0.3, 0.3, num=6))
plt.xlabel('x', fontstyle='italic', fontsize=15)
plt.ylabel('y', fontstyle='italic', fontsize=15)
plt.title('Abs Error(U-Velocity)', fontfamily='Times New Roman', fontsize=15, fontweight='bold', fontstyle='italic')
plt.savefig(r"F:\data\2D\error-u.tif", dpi=300, format="tiff",
            bbox_inches="tight", pad_inches=0.1)
plt.show()


# Plot the error histogram
mean_error = np.nanmean(err_used)
error = np.where(np.isnan(err_used), mean_error, err_used)

abs_error_values = error.flatten()
abs_error_values = abs_error_values[np.isfinite(abs_error_values)]

# Fixed bin width
x_min, x_max = 0.0, 0.01
n_bins = 40
bin_w = (x_max - x_min) / n_bins
bins = np.arange(x_min, x_max + bin_w/2, bin_w)  # 等间隔边界（含右端点）

fig, ax = plt.subplots(1, 1, figsize=(12, 5))
plt.rcParams['font.family'] = 'Times New Roman'

ax.hist(abs_error_values, bins=bins, color='orange', edgecolor='black', alpha=0.7)
ax.set_xlabel(r"Absolute Error Value(U-Velocity)", fontfamily='Times New Roman',fontsize=15, fontweight='bold', fontstyle='italic')
ax.set_ylabel("Frequency", fontfamily='Times New Roman',fontsize=15, fontweight='bold', fontstyle='italic')
ax.set_xlim(0, 0.01)
ax.set_ylim(0, 8000)
plt.savefig(r"F:\data\2D\histogram_error_v.tif", dpi=300, format="tiff",
            bbox_inches="tight", pad_inches=0.1)
plt.show()


