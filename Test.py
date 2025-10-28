import os
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn.functional as F
from model import SnoutNet
from PretrainedModel import build_model
from torchvision import transforms
from typing import List, Tuple
from oxfordDataset import OxfordPetNoses
from torch.utils.data import DataLoader, random_split


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


# ==== Ensemble helpers (paste once) ==========================================
def _build_model_by_name(name: str):
    name = name.lower()
    if name == "snoutnet":
        from model import SnoutNet
        return SnoutNet()
    elif name in ("alexnet", "vgg16"):
        # Your pretrained wrapper that returns a 2-dim head
        from PretrainedModel import build_model
        return build_model(name, pretrained=False)
    raise ValueError(f"Unknown backbone: {name}")


def _load_state_safely(model, path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)


def _parse_ensemble_arg(ens: str) -> List[Tuple[str, str]]:
    # "alexnet:a.pth,vgg16:b.pth,snoutnet:c.pth" → [("alexnet","a.pth"), ...]
    if not ens: return []
    out = []
    for part in ens.split(","):
        if not part.strip(): continue
        b, ckpt = part.split(":", 1)
        out.append((b.strip(), ckpt.strip()))
    return out


@torch.no_grad()
def _ensemble_predict(models: List[torch.nn.Module], weights: torch.Tensor, imgs: torch.Tensor) -> torch.Tensor:
    # imgs: [B,3,H,W]; returns [B,2]
    preds = torch.stack([m(imgs) for m in models], dim=0)  # [M,B,2]
    w = (weights / weights.sum()).view(-1, 1, 1)           # [M,1,1]
    return (preds * w).sum(dim=0)


def _optimize_ens_weights(models, device, args):
    """
    Learn non-negative weights (softmax) on a held-out VAL portion of the TRAIN set.
    No augmentation is used here.
    """
    # Build a small validation loader from your training split
    train_ds = OxfordPetNoses(
        root_dir=args.data_root,
        mode="train",
        img_size=args.img_size,
        keep_aspect=getattr(args, "keep_aspect", False),
        transform=None,
        normalize=True
    )

    # Take a fraction as validation (default 20%)
    val_frac = getattr(args, "ens_val_fraction", 0.2)
    n_val = max(1, int(len(train_ds) * val_frac))
    n_tr  = max(1, len(train_ds) - n_val)
    _, val_ds = random_split(train_ds, [n_tr, n_val],
                             generator=torch.Generator().manual_seed(123))

    val_loader = DataLoader(val_ds, batch_size=getattr(args, "batch_size", 32),
                            shuffle=False, num_workers=0, pin_memory=False)

    # Collect predictions of each model on VAL
    preds_all, targets = [], []
    with torch.no_grad():
        for imgs, uv in val_loader:
            imgs = imgs.to(device).float()
            targets.append(uv.to(device).float())             # [B,2]
            preds_all.append(torch.stack([m(imgs) for m in models], dim=0))  # [M,B,2]

    if len(preds_all) == 0:
        # fall back to equal weights if something went wrong
        return torch.ones(len(models), dtype=torch.float32, device=device)

    preds_all = torch.cat(preds_all, dim=1)  # [M,N,2]
    targets   = torch.cat(targets,   dim=0)  # [N,2]

    # Learn weights on simplex via softmax(logits)
    M = preds_all.size(0)
    w_logits = torch.zeros(M, device=device, requires_grad=True)
    opt = torch.optim.Adam([w_logits], lr=getattr(args, "ens_lr", 0.1))
    mse = torch.nn.MSELoss()

    steps = getattr(args, "ens_steps", 300)
    for _ in range(steps):
        w = F.softmax(w_logits, dim=0)               # >=0 and sums to 1
        ens = (preds_all * w.view(M,1,1)).sum(dim=0) # [N,2]
        loss = mse(ens, targets)
        opt.zero_grad(); loss.backward(); opt.step()

    w = F.softmax(w_logits, dim=0).detach()
    print("Optimized ensemble weights:", [round(float(x), 4) for x in w])
    return w
# =====================================================================


def _denorm_img(x):
    """x: [3,H,W] tensor (raw [0,1] or ImageNet-normalized) -> HxWx3 float [0,1]."""
    x = x.detach().cpu()
    if x.ndim != 3:
        raise ValueError("expected a single image tensor [3,H,W]")
    # heuristic: if it looks normalized, de-normalize
    if x.mean() < 0 or x.max() > 1.5:
        dn = transforms.Normalize(
            mean=[-m/s for m, s in zip(_IMAGENET_MEAN, _IMAGENET_STD)],
            std=[1/s for s in _IMAGENET_STD],
        )
        x = dn(x)
    x = x.clamp(0, 1)
    return x.permute(1, 2, 0).numpy()  # HWC


def visualize_prediction(img, pred_uv, gt_uv=None, save_path=None, title=None, point_radius=5):
    """
    img:      tensor [3,H,W] (raw or ImageNet-normalized)
    pred_uv:  [2] (u,v) in pixels
    gt_uv:    optional [2] (u,v) in pixels
    """
    im = _denorm_img(img)
    H, W = im.shape[:2]
    u, v = float(pred_uv[0]), float(pred_uv[1])
    u = max(0.0, min(u, W - 1.0))
    v = max(0.0, min(v, H - 1.0))

    plt.figure(figsize=(3.0, 3.0), dpi=120)
    plt.imshow(im)
    plt.axis("off")

    # purple prediction dot
    plt.scatter([u], [v], s=point_radius**2, c=[(0.6, 0.0, 0.8)], zorder=5)

    # red u (left) and v (up) arrows from prediction point
    plt.annotate("", xy=(0, v), xytext=(u, v),
                 arrowprops=dict(arrowstyle="->", color="red", lw=2))
    plt.annotate("", xy=(u, 0), xytext=(u, v),
                 arrowprops=dict(arrowstyle="->", color="red", lw=2))

    # optional GT (green)
    if gt_uv is not None:
        gu = max(0.0, min(float(gt_uv[0]), W - 1.0))
        gv = max(0.0, min(float(gt_uv[1]), H - 1.0))
        plt.scatter([gu], [gv], s=point_radius**2, c=["lime"], edgecolors="black",
                    linewidths=0.5, zorder=6)

    if title:
        plt.title(title, fontsize=9)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.tight_layout(pad=0)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()
    else:
        plt.show()
# --- END: viz helper ---


def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():          return torch.device("cuda")
    return torch.device("cpu")


def make_loader(data_root, img_size, batch_size):
    tx = transforms.Compose([
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    ds = OxfordPetNoses(root_dir=data_root,mode="test", keep_aspect=True, transform=tx, normalize=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=False)


def evaluate(model, loader, device, return_errors=False):
    model.eval()
    all_err = []

    with torch.no_grad():
        for imgs, uv in loader:
            imgs = imgs.to(device).float()
            uv   = uv.to(device).float()
            preds = model(imgs)

            diff = preds - uv                 # (B,2)
            e = torch.sqrt((diff**2).sum(dim=1))  # (B,) Euclidean px error
            all_err.append(e.cpu())

    err = torch.cat(all_err)                  # (N,)
    out = {
        "mean": err.mean().item(),
        "std":  err.std(unbiased=False).item(),
        "min":  err.min().item(),
        "max":  err.max().item(),
    }
    return (out, err) if return_errors else out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--ckpt", type=str, default="non-aug_snoutnet_best.pth")
    ap.add_argument("--data_root", type=str, default="data/oxford-iiit-pet-noses")
    ap.add_argument("--img_size", type=int, default=227)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("-t", "--toggle_pretrain", type=int, default=0)
    ap.add_argument("--backbone", type=str, default="alexnet", choices=["alexnet", "vgg16"])
    ap.add_argument("--viz", action="store_true", help="save a few prediction visualizations")
    ap.add_argument("--viz-count", type=int, default=1, help="number of samples to visualize")
    ap.add_argument("--viz-dir", type=str, default="viz", help="output folder for saved visualizations")

    # --- Ensemble options ---
    ap.add_argument(
        "--ensemble",
        type=str,
        default="",
        help=("Comma list of backbone:ckpt pairs. "
              "Backbones: snoutnet|alexnet|vgg16. "
              "Example: 'snoutnet:snoutnet_best.pth,alexnet:alexA.pth,vgg16:vggA.pth'")
    )
    ap.add_argument(
        "--ens-weights",
        type=str,
        default="",
        help="Optional comma list of weights matching --ensemble, e.g. '1,1,2'. Defaults to equal."
    )

    return ap.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"device: {device}")

    test_loader = make_loader(args.data_root, args.img_size, args.batch_size)

    if args.toggle_pretrain==1:
        model = build_model(args.backbone, pretrained=True, freeze_backbone=True)
        model = model.to(device).float()
    else:
        model = SnoutNet().to(device).float()

    state = torch.load(args.ckpt, map_location=device)
    # support both state dict or wrapped dict
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)

    stats = evaluate(model, test_loader, device)
    print(f"Euclidean px error → mean={stats['mean']:.2f}, "
          f"std={stats['std']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}")

    # --- OPTIONAL: save visualizations (keeps your original code intact) ---
    if args.viz:
        model.eval()
        saved = 0
        with torch.no_grad():
            for imgs, uv in test_loader:  # reuse your existing 'loader' and 'model' variables
                imgs = imgs.to(device).float()
                preds = model(imgs)
                for i in range(min(args.viz_count - saved, imgs.size(0))):
                    visualize_prediction(
                        img=imgs[i],
                        pred_uv=preds[i],
                        gt_uv=uv[i],  # or None if you only want the guess
                        save_path=os.path.join(args.viz_dir, f"viz_{saved:04d}.png"),
                        title=f"pred={preds[i].tolist()}  gt={uv[i].tolist()}"
                    )
                    saved += 1
                    if saved >= args.viz_count:
                        break
                if saved >= args.viz_count:
                    break
        print(f"saved {saved} visualization(s) to {args.viz_dir}")

    # ---- OPTIONAL: visualize prediction on an AUGMENTED image (flip/jitter/rot etc.) ----
    if args.viz:
        # Build a temporary TRAIN dataset that applies your augmentations.
        # We set use_aug=True; if your dataset exposes aug_hflip_p / aug_rot90 / color jitter,
        # we pass them when available without breaking older versions.
        ds_kwargs = dict(
            root_dir=args.data_root,
            mode="train",
            img_size=args.img_size,
            keep_aspect=getattr(args, "keep_aspect", False),
            transform=None,
            normalize=True,
            aug_rotate=True,
            aug_hflip_p=0.5,
            aug_color_jitter={"brightness": 0.2, "contrast": 0.2, "saturation": 0.2},
            show_aug_once=True,
            show_aug_path="aug_example.png"
        )
        # Optional knobs (only used if your dataset supports them)
        if hasattr(OxfordPetNoses, "__init__"):
            try:
                ds_kwargs["aug_hflip_p"] = 0.5  # flip probability
                ds_kwargs["aug_rotate"] = True  # enable 0/90/180/270 rotations
                ds_kwargs["aug_color_jitter"] = {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2}
            except Exception:
                pass

        aug_train = OxfordPetNoses(**ds_kwargs)
        aug_loader = DataLoader(aug_train, batch_size=1, shuffle=True,
                                num_workers=0, pin_memory=False)

        model.eval()
        with torch.no_grad():
            imgs, _ = next(iter(aug_loader))  # labels intentionally ignored
            imgs = imgs.to(device).float()
            preds = model(imgs).cpu()
            out_path = os.path.join(args.viz_dir, "aug_pred.png")
            visualize_prediction(imgs[0].cpu(), preds[0], gt_uv=None, save_path=out_path)
        print("Saved augmented prediction to", os.path.abspath(out_path))

if __name__ == "__main__":
    main()
