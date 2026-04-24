import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

def _is_img(p):
    return p.lower().endswith(IMG_EXTS)

def _stem(name):
    base = os.path.splitext(os.path.basename(name))[0]
    return base

def _normalize_key(stem):
    """
    For RESIDE-like hazy names: 'abc_0.8_0.2' -> 'abc'
    Also handles: 'abc_1' etc. If no underscore, returns stem.
    """
    if "_" in stem:
        return stem.split("_")[0]
    return stem

class PairedDehazeDataset(Dataset):
    """
    Expects:
      root/
        hazy/   (hazy images)
        clear/  (ground-truth clear images)
    Pairs images by matching stem keys.
    """
    def __init__(self, root, split="train", val_ratio=0.1, seed=42,
                 size=256, random_crop=True):
        super().__init__()
        self.hazy_dir = os.path.join(root, "hazy")
        self.clear_dir = os.path.join(root, "clear")

        hazy_paths = sorted([p for p in glob(os.path.join(self.hazy_dir, "*")) if _is_img(p)])
        clear_paths = sorted([p for p in glob(os.path.join(self.clear_dir, "*")) if _is_img(p)])

        clear_map = {}
        for p in clear_paths:
            clear_map[_stem(p)] = p  # exact stem
            clear_map[_normalize_key(_stem(p))] = p  # also normalized

        pairs = []
        for hp in hazy_paths:
            hs = _stem(hp)
            key1 = hs
            key2 = _normalize_key(hs)
            cp = clear_map.get(key1, None)
            if cp is None:
                cp = clear_map.get(key2, None)
            if cp is not None:
                pairs.append((hp, cp))

        if len(pairs) == 0:
            raise RuntimeError(
                f"No pairs found. Check structure:\n"
                f"  {self.hazy_dir}\n  {self.clear_dir}\n"
                f"and ensure filenames can be matched."
            )

        # deterministic split
        import random
        random.Random(seed).shuffle(pairs)
        n_val = int(len(pairs) * val_ratio)
        if split == "train":
            self.pairs = pairs[n_val:]
        else:
            self.pairs = pairs[:n_val]

        # transforms
        # For dehazing, photometric augmentations can distort the haze physics;
        # keep it conservative.
        self.size = size
        self.random_crop = random_crop

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.pairs)

    def _resize_or_crop(self, img):
        # img: PIL
        w, h = img.size
        # ensure min size
        if min(w, h) < self.size:
            scale = self.size / min(w, h)
            nw, nh = int(w * scale), int(h * scale)
            img = img.resize((nw, nh), Image.BICUBIC)

        if self.random_crop:
            # random crop
            import random
            w, h = img.size
            x1, y1 = 0, 0
            
            if w == self.size and h == self.size:
                return img
            
            if w > self.size:
                x1 = random.randint(0, w - self.size)
            
            if h > self.size:
                y1 = random.randint(0, h - self.size)
            
            img = img.crop((x1, y1, x1 + self.size, y1 + self.size))
        else:
            # center crop
            img = T.CenterCrop(self.size)(img)
        return img

    def __getitem__(self, idx):
        hp, cp = self.pairs[idx]
        hazy = Image.open(hp).convert("RGB")
        clear = Image.open(cp).convert("RGB")

        hazy = self._resize_or_crop(hazy)
        clear = self._resize_or_crop(clear)

        # simple geometric augmentation (keep consistent)
        import random
        if random.random() < 0.5:
            hazy = T.functional.hflip(hazy)
            clear = T.functional.hflip(clear)
        if random.random() < 0.5:
            hazy = T.functional.vflip(hazy)
            clear = T.functional.vflip(clear)

        hazy_t = self.to_tensor(hazy)   # [0,1]
        clear_t = self.to_tensor(clear) # [0,1]

        return {
            "hazy": hazy_t,
            "clear": clear_t,
            "hazy_path": hp,
            "clear_path": cp
        }