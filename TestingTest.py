# test.py — inference-only (no training), supports single model or ensemble
# Examples:
#   python test.py -c snoutnet_best.pth
#   python test.py --backbone alexnet -c alex_A.pth
#   python test.py --ensemble "snoutnet:snoutnet_best.pth,alexnet:alex_A.pth,vgg16:vgg_A.pth"
#   python test.py -c snoutnet_best.pth --viz --viz-count 4
#   python test.py --ensemble "snoutnet:a.pth,alexnet:b.pth" --viz-aug

import os
import math
import argparse
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# ---------------- Device ----------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# ---------------- Dataset ----------------
# Your dataset class (expects these args)
from oxfordDataset import OxfordPetNoses

# ---------------- Visualization helpers ----------------
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

def _denorm_img(x):
    """x: [3,H,W] tensor (raw [0,1] or ImageNet-normalized) -> HWC float [0,1]."""
    x = x.detach().cpu()
    # heuristic: if looks normalized, de-normalize
    if x.mean() < 0 or x.max() > 1.5:
        dn = transforms.Normalize(
            mean=[-m/s for m, s in zip(_IMAGENET_MEAN, _IMAGENET_STD)],
            std=[1/s for s in _IMAGENET_STD],
        )
        x = dn(x)
    return x.clamp(0, 1).permute(1, 2, 0).numpy()

def visualize_prediction(img, pred_uv, gt_uv=None, save_path=None, title=None, point_radius=5):
    """Draw prediction (purple) with u/v arrows (red). Optionally draw GT (green)."""
    im = _denorm_img(img)
    H, W = im.shape[:2]

    u, v = float(pred_uv[0]), float(pred_uv[1])
    u = max(0.0, min(u, W - 1.0))
    v = max(0.0, min(v, H - 1.0))

    plt.figure(figsize=(3.2, 3.2), dpi=120)
    plt.imshow(im)
    plt.axis("off")

    # prediction = purple
    plt.scatter([u], [v], s=point_radius**2, c=[(0.6, 0.0, 0.8)], zorder=5)

    # u (left) and v (up) arrows from prediction point
    plt.annotate("", xy=(0, v), xytext=(u, v), arrowprops=dict(arrowstyle="->", color="red", lw=2))
    plt.annotate("", xy=(u, 0), xytext=(u, v), arrowprops=dict(arrowstyle="->", color="red", lw=2))

    # ground truth = green (optional)
    if gt_uv is not None:
        gu = max(0.0, min(float(gt_uv[0]), W - 1.0))
        gv = max(0.0, min(float(gt_uv[1]), H - 1.0))
        plt.scatter([gu], [gv], s=point_radius**2, c=["lime"], edgecolors="black", linewidths=0.5, zorder=6)

    if title:
        plt.title(title, fontsize=9)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.tight_layout(pad=0)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()
    else:
        plt.show()

def visualize_prediction_only(img, pred_uv, save_path):
    """Prediction only (purple dot + u/v arrows)."""
    visualize_prediction(img=img, pred_uv=pred_uv, gt_uv=None, save_path=save_path, title=None)

# ---------------- Ensemble helpers (inference-only) ----------------
def _build_model_by_name(name: str):
    name = name.lower()
    if name == "snoutnet":
        from model import SnoutNet
        return SnoutNet()
    elif name in ("alexnet", "vgg16"):
        # Your wrapper that returns a 2-dim head
        from PretrainedModel import build_model
        return build_model(name, pretrained=False)
    raise ValueError(f"Unknown backbone: {name}")

def _load_state_safely(model, path, device):
    state = torch.load(path, map_location=device)
    # allow {"state_dict": ...} or {"model": ...} wrappers
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    model.load_state_dict(state, strict=True)

def _parse_ensemble_arg(ens: str) -> List[Tuple[str, str]]:
    # "alexnet:a.pth,vgg16:b.pth,snoutnet:c.pth" → [("alexnet","a.pth"), ...]
    if not ens:
        return []
    out = []
    for part in ens.split(","):
        part = part.strip()
        if not part:
            continue
        b, ckpt = part.split(":", 1)
        out.append((b.strip(), ckpt.strip()))
    return out

@torch.no_grad()
def _ensemble_predict(models: List[torch.nn.Module], weights: torch.Tensor, imgs: torch.Tensor) -> torch.Tensor:
    """imgs: [B,3,H,W]; returns [B,2]."""
    preds = torch.stack([m(imgs) for m in models], dim=0)  # [M,B,2]
    w = (weights / weights.sum()).view(-1, 1, 1)           # [M,1,1]
    return (preds * w).sum(dim=0)

def _weights_from_arg(arg: str, n: int, device):
    if not arg:
        return torch.ones(n, dtype=torch.float32, device=device)
    vals = [float(x.strip()) for x in arg.split(",") if x.strip()]
    if len(vals) != n:
        raise ValueError(f"--ens-weights must have {n} numbers (got {len(vals)}).")
    return torch.tensor(vals, dtype=torch.float32, device=device)

# ---------------- Argparse ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--ckpt", type=str, default="", help="checkpoint path (single model mode)")
    ap.add_argument("--backbone", type=str, default="snoutnet", choices=["snoutnet", "alexnet", "vgg16"])
    ap.add_argument("--data_root", type=str, default="data/oxford-iiit-pet-noses")
    ap.add_argument("--img_size", type=int, default=227)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--keep_aspect", action="store_true", help="letterbox pad to square instead of stretching")

    # Visualizations (non-augmented test samples)
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--viz-count", type=int, default=4)
    ap.add_argument("--viz-dir", type=str, default="viz")

    # Visualize on an augmented TRAIN sample (prediction only)
    ap.add_argument("--viz-aug", action="store_true")
    ap.add_argument("--viz-aug-dir", type=str, default="viz_aug")

    # Ensemble (inference-only; no optimization here)
    ap.add_argument("--ensemble", type=str, default="",
                    help="comma list of backbone:ckpt pairs, e.g. 'snoutnet:a.pth,alexnet:b.pth'")
    ap.add_argument("--ens-weights", type=str, default="",
                    help="manual weights (comma list) matching --ensemble; if empty, equal weights")
    return ap.parse_args()

# ---------------- Evaluation ----------------
@torch.no_grad()
def evaluate_any(predict_fn, loader, device):
    """predict_fn(imgs)->preds; returns (mse, mean, std, min, max) for Euclidean px error."""
    loss_fn = nn.MSELoss()
    n = len(loader.dataset)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    mse_acc = 0.0
    all_e = []

    for imgs, uv in loader:
        imgs = imgs.to(device).float()
        uv   = uv.to(device).float()         # [B,2]
        preds = predict_fn(imgs)             # [B,2]

        mse_acc += loss_fn(preds, uv).item() * imgs.size(0)

        diff = preds - uv
        e = torch.sqrt((diff ** 2).sum(dim=1))  # Euclidean per sample [B]
        all_e.append(e.cpu())

    mse = mse_acc / n
    e = torch.cat(all_e)
    return mse, e.mean().item(), e.std(unbiased=False).item(), e.min().item(), e.max().item()

# ---------------- Main ----------------
def main():
    args = parse_args()
    device = get_device()
    print("device:", device.type)

    # Build test loader (no augmentation)
    test_set = OxfordPetNoses(
        root_dir=args.data_root,
        mode="test",
        img_size=args.img_size,
        keep_aspect=args.keep_aspect,
        transform=None,
        normalize=True
    )
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=False)

    # ---- Build model(s) ----
    pairs = _parse_ensemble_arg(args.ensemble)
    use_ensemble = len(pairs) > 0

    if use_ensemble:
        models = []
        for backbone, ckpt in pairs:
            m = _build_model_by_name(backbone).to(device).float()
            _load_state_safely(m, ckpt, device)
            m.eval()
            models.append(m)

        # Inference-only: ignore any optimization flags; use manual or equal weights
        weights = _weights_from_arg(args.ens_weights, len(models), device)

        def _predict_any(imgs):
            return _ensemble_predict(models, weights, imgs)

    else:
        # single-model path (same behavior you had before)
        if args.backbone == "snoutnet":
            from model import SnoutNet
            model = SnoutNet().to(device).float()
        else:
            from PretrainedModel import build_model
            model = build_model(args.backbone, pretrained=False).to(device).float()

        if not args.ckpt:
            raise ValueError("Please provide -c/--ckpt for single-model evaluation.")
        state = torch.load(args.ckpt, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        model.load_state_dict(state, strict=True)
        model.eval()

        def _predict_any(imgs):
            return model(imgs)

    # ---- Evaluate ----
    mse, mean_e, std_e, min_e, max_e = evaluate_any(_predict_any, loader, device)
    print(f"val_loss(MSE per coord) = {mse:.4f}")
    print(f"Euclidean px error → mean={mean_e:.2f}, std={std_e:.2f}, min={min_e:.2f}, max={max_e:.2f}")

    # ---- Optional: visualize a few test samples (non-augmented) ----
    if args.viz:
        os.makedirs(args.viz_dir, exist_ok=True)
        with torch.no_grad():
            imgs, uv = next(iter(loader))
            imgs = imgs.to(device).float()
            preds = _predict_any(imgs).cpu()
            show_n = min(args.viz_count, imgs.size(0))
            for i in range(show_n):
                out_path = os.path.join(args.viz_dir, f"viz_test_{i:04d}.png")
                visualize_prediction(img=imgs[i].cpu(), pred_uv=preds[i], gt_uv=uv[i], save_path=out_path)
        print(f"saved {show_n} visualization(s) to {os.path.abspath(args.viz_dir)}")

    # ---- Optional: visualize ONE augmented TRAIN sample (prediction only) ----
    if args.viz_aug:
        os.makedirs(args.viz_aug_dir, exist_ok=True)
        # Build a tiny TRAIN dataset that applies your augmentations
        ds_kwargs = dict(
            root_dir=args.data_root,
            mode="train",
            img_size=args.img_size,
            keep_aspect=args.keep_aspect,
            transform= transforms.Compose([
                        transforms.Resize((227, 227), antialias=True),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],
                                                [0.229, 0.224, 0.225]),
            ]),
            normalize=True,
            aug_rotate=True,
            aug_hflip_p=0.5,
            aug_color_jitter={"brightness": 0.2, "contrast": 0.2, "saturation": 0.2},
            show_aug_once=True,
            show_aug_path="aug_example.png"
        )
        # If your dataset exposes explicit knobs, you can add them here:
        # ds_kwargs["aug_hflip_p"] = 0.5
        # ds_kwargs["aug_rot90"]   = True
        # ds_kwargs["aug_color_jitter"] = {"brightness":0.2,"contrast":0.2,"saturation":0.2}

        aug_train = OxfordPetNoses(**ds_kwargs)
        aug_loader = DataLoader(aug_train, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)

        with torch.no_grad():
            imgs, _ = next(iter(aug_loader))  # labels ignored intentionally
            imgs = imgs.to(device).float()
            preds = _predict_any(imgs).cpu()
            out_path = os.path.join(args.viz_aug_dir, "aug_pred.png")
            visualize_prediction_only(img=imgs[0].cpu(), pred_uv=preds[0], save_path=out_path)
        print(f"saved augmented prediction to {os.path.abspath(out_path)}")

if __name__ == "__main__":
    main()
