# oxfordDataset.py
import os, re, random, torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as F


class OxfordPetNoses(Dataset):
    """
    Oxford-IIIT Pet Noses (ELEC475-style)
    Keeps original arg names: mode, keep_aspect, transform.
    """
    def __init__(self,
                 root_dir="data/oxford-iiit-pet-noses",
                 mode="train",                 # <-- original
                 img_size=227,
                 keep_aspect=False,            # <-- original
                 transform=None,               # <-- original (photometric-only recommended)
                 normalize=True,
                 aug_rotate: bool = False,
                 aug_hflip_p: float = 0.0,
                 aug_color_jitter: dict | None = None,
                 show_aug_once: bool = False,
                 show_aug_path: str = "aug_example.png"
                 ):

        assert mode in ("train", "test")
        self.root_dir = root_dir
        self.mode = mode
        self.img_size = int(img_size)
        self.keep_aspect = bool(keep_aspect)
        self.transform = transform           # user-supplied transforms (applied last)
        self.normalize = bool(normalize)

        # Aug configs
        self.aug_rotate = bool(aug_rotate)
        self.aug_hflip_p = float(aug_hflip_p)
        self.cj = T.ColorJitter(**aug_color_jitter) if aug_color_jitter else None


        self.use_aug = bool(aug_rotate)

        # one-time viz
        self.show_aug_once = bool(show_aug_once)
        self.show_aug_path = show_aug_path
        self._shown_aug = False

        # Files
        self.img_dir = os.path.join(root_dir, "images-original", "images")
        ann = "train_noses.txt" if mode == "train" else "test_noses.txt"
        with open(os.path.join(root_dir, ann), "r") as f:
            self.lines = [ln.strip() for ln in f if ln.strip()]
        self._pat = re.compile(r'(.+?),"\((\d+),\s*(\d+)\)"')

        # Defaults if no user transform is provided
        self.to_tensor = T.ToTensor()
        self.normalize_tf = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ) if self.normalize else None

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        m = self._pat.match(self.lines[idx])
        if not m:
            raise ValueError(f"Bad annotation line: {self.lines[idx]}")
        fname, u_str, v_str = m.groups()
        u_raw, v_raw = float(u_str), float(v_str)

        path = os.path.join(self.img_dir, fname)
        img = Image.open(path).convert("RGB")
        w, h = img.size

        if not self.keep_aspect:
            # --- Stretch to square ---
            sx, sy = self.img_size / w, self.img_size / h
            u, v = u_raw * sx, v_raw * sy
            img = img.resize((self.img_size, self.img_size), resample=Image.BICUBIC)
        else:
            # --- Letterbox (keep aspect, pad to square) ---
            scale = self.img_size / min(w, h)
            new_w, new_h = round(w * scale), round(h * scale)
            img = img.resize((new_w, new_h), resample=Image.BICUBIC)
            pad_l = (self.img_size - new_w) // 2
            pad_t = (self.img_size - new_h) // 2
            pad_r = self.img_size - new_w - pad_l
            pad_b = self.img_size - new_h - pad_t
            img = F.pad(img, (pad_l, pad_t, pad_r, pad_b), fill=0)
            u, v = u_raw * scale + pad_l, v_raw * scale + pad_t

        # --------- Augmentations (train only) ----------
        if self.mode == "train":

            # Horizontal flip (geometric → update u)
            if self.aug_hflip_p > 0.0 and random.random() < self.aug_hflip_p:
                img = F.hflip(img)
                u = self.img_size - 1 - u

            # Color jitter (photometric → labels unchanged)
            if self.cj is not None:
                img = self.cj(img)

            # Rotations by 0/90/180/270° (counter-clockwise)
            if self.aug_rotate:
                k = random.choice([0, 1, 2, 3])  # 0:0°, 1:90°, 2:180°, 3:270° CCW
                if k != 0:
                    img = F.rotate(img, angle=90 * k)  # CCW rotation
                    S = self.img_size - 1  # index max
                    # Update (u, v) for CCW 90° rotations around the image center:
                    if k == 1:  # 90° CCW
                        u, v = v, (S - u)
                    elif k == 2:  # 180°
                        u, v = (S - u), (S - v)
                    elif k == 3:  # 270° CCW (or 90° CW)
                        u, v = (S - v), u

        # --------- To tensor (+ optional normalize) ----------
        if self.transform is not None:
            # NOTE: your `transform` should be photometric-only (no geometry),
            # because (u,v) have already been adjusted above.
            img = self.transform(img)
        else:
            img = self.to_tensor(img)
            if self.normalize_tf is not None:
                img = self.normalize_tf(img)

        label = torch.tensor([u, v], dtype=torch.float32)
        return img, label
