#!/usr/bin/env python
# CatOrDog/classifier.py
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Iterable, Dict, List, Literal

import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torchvision import models
from tqdm import tqdm

Label = Literal["cat", "dog", "unknown"]


# ─────────────────────────── chargeur de backbone (tri principal)
def _load_backbone(name: str, device: str):
    logging.info("⏳ Chargement modèle %s …", name)
    name = name.lower()
    if name == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        net = models.efficientnet_b0(weights=w)
    elif name == "convnext_tiny":
        w = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        net = models.convnext_tiny(weights=w)
    else:  # resnet50 par défaut
        w = models.ResNet50_Weights.IMAGENET1K_V2
        net = models.resnet50(weights=w)

    net.eval().to(device)
    return net, w.transforms(), w.meta["categories"]


# ─────────────────────────── classifieur principal
class CatDogClassifier:
    """Chaîne multi-modèles (ResNet-50 ➜ EfficientNet-B0 ➜ ConvNeXt-Tiny + raffinage MobileNet)."""

    SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
    _CACHE: dict[str, tuple[torch.nn.Module, object, list[str]]] = {}

    def __init__(
        self,
        models_order: List[str] | None = None,
        device: str | None = None,
        batch_size: int = 64,
    ):
        self.names = models_order or ["resnet50", "efficientnet_b0", "convnext_tiny"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch = max(1, batch_size)

        self.models: list[tuple[torch.nn.Module, object, list[str]]] = []
        for name in self.names:
            if name not in self._CACHE:
                self._CACHE[name] = _load_backbone(name, self.device)
            self.models.append(self._CACHE[name])

    # ─────────── prédiction d’une image   ────────────────────────────────────
    def predict_image(self, path: Path) -> Label:
        if path.suffix.lower() not in self.SUPPORTED_EXT:
            raise ValueError(f"Extension non gérée : {path.suffix}")
        try:
            img = Image.open(path).convert("RGB")
        except UnidentifiedImageError:
            logging.warning("Illisible : %s", path.name)
            return "unknown"

        for net, tfm, labels in self.models:
            x = tfm(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                idx = net(x).argmax(1).item()
            lab = self._idx_to_label(idx, labels)
            if lab != "unknown":
                return lab
        return "unknown"

    # ─────────── prédiction d’un lot - passes successives ───────────────────
    def classify_iter(self, files: Iterable[Path]) -> Dict[Path, Label]:
        files = list(files)
        remaining = files
        results: Dict[Path, Label] = {}

        for net, tfm, labels in self.models:
            if not remaining:
                break
            new_remaining = []
            for i in tqdm(
                range(0, len(remaining), self.batch),
                desc=f"→ {net.__class__.__name__}",
                unit="img",
            ):
                subset = remaining[i : i + self.batch]
                imgs, keep = [], []
                for p in subset:
                    try:
                        imgs.append(tfm(Image.open(p).convert("RGB")))
                        keep.append(p)
                    except UnidentifiedImageError:
                        results[p] = "unknown"

                if not imgs:
                    continue

                preds = net(torch.stack(imgs).to(self.device)).argmax(1).cpu()
                for p, idx in zip(keep, preds):
                    lab = self._idx_to_label(int(idx), labels)
                    if lab == "unknown":
                        new_remaining.append(p)
                    else:
                        results[p] = lab
            remaining = new_remaining

        # ce qui reste : unknown
        for p in remaining:
            results[p] = "unknown"
        return results

    # ─────────── passe unique (copie ou déplacement) ─────────────────────────
    def classify_once(
        self,
        src: Path,
        dst: Path,
        *,
        move_src: bool,
        keep_unknown: bool = True,
    ) -> int:
        """Trie *src* → *dst* et renvoie le nombre de fichiers restés unknown."""
        files = [
            f
            for f in src.iterdir()
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXT
        ]
        if not files:
            return 0

        res = self.classify_iter(files)
        unknown = 0
        for p, lab in res.items():
            if lab == "unknown":
                if not keep_unknown:
                    continue
                dst_file = dst / "unknow" / p.name
                unknown += 1
            else:
                dst_file = dst / lab / p.name
            self._move(p, dst_file, mv=move_src)
        return unknown

    # ─────────── boucle complète + raffinage MobileNet ──────────────────────
    def classify_loop(
        self,
        src: Path,
        dst: Path,
        *,
        copy_src: bool = True,
        max_passes: int = 10,
        verbose_refine: bool = False,
    ):
        """Tri complet : passes classiques puis raffinage MobileNet forcé."""
        self.classify_once(src, dst, move_src=not copy_src)

        # passes classiques sur unknow/
        for n in range(1, max_passes + 1):
            unk_dir = dst / "unknow"
            if not any(unk_dir.iterdir()):
                logging.info("✅ plus rien dans 'unknow/'")
                break
            logging.info("🔄 Passe %d sur 'unknow/'", n)
            before = sum(1 for _ in unk_dir.iterdir())
            after_unknown = self.classify_once(
                unk_dir,
                dst,
                move_src=True,
                keep_unknown=True,
            )
            if after_unknown == before:  # plus aucun progrès
                break

        # raffinage final : 100 % des fichiers encore unknown sont rangés
        moved, _ = self.refine_unknown(dst, verbose=verbose_refine)
        if moved:
            logging.info("📝 Raffinage MobileNet : %d fichier(s) rangés", moved)

    # ─────────── purge dossiers ──────────────────────────────────────────────
    @staticmethod
    def clear(root: Path, *, remove_dirs=False):
        for sub in ("cat", "dog", "unknow"):
            folder = root / sub
            if folder.exists():
                shutil.rmtree(folder)
                if not remove_dirs:
                    folder.mkdir(parents=True, exist_ok=True)

    # ─────────── raffinage « pourcentage » MobileNet (ex-re_tri) ────────────
    def refine_unknown(
        self,
        root: Path,
        *,
        batch_size: int | None = None,
        verbose: bool = False,
        move: bool = True,
    ) -> tuple[int, list[tuple[str, float, float, str]]]:
        """
        Classe toutes les images du dossier <root>/unknow/ via MobileNet V3.
        Attribution forcée : DOG si p(dog) ≥ p(cat) sinon CAT.
        Retour : (nb_déplacées, détails).
        """
        unk_dir = root / "unknow"
        if not unk_dir.exists():
            return 0, []

        remaining = [
            p
            for p in unk_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXT
        ]
        if not remaining:
            return 0, []

        # chargement / cache MobileNet V3 Large
        key = "mv3_large"
        if key not in self._CACHE:
            logging.info("⏳ Chargement MobileNetV3-Large …")
            w = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
            self._CACHE[key] = (
                models.mobilenet_v3_large(weights=w).eval().to(self.device),
                w.transforms(),
                w.meta["categories"],
            )
        mv3, mv3_tfm, mv3_labels = self._CACHE[key]

        cat_ids = [i for i, l in enumerate(mv3_labels) if "cat" in l.lower()]
        dog_ids = [
            i
            for i, l in enumerate(mv3_labels)
            if any(
                k in l.lower()
                for k in (
                    "dog",
                    "hound",
                    "terrier",
                    "retriever",
                    "shepherd",
                    "spaniel",
                    "collie",
                    "poodle",
                    "husky",
                    "malamute",
                    "beagle",
                )
            )
        ]

        bs = batch_size or self.batch
        moved = 0
        details: list[tuple[str, float, float, str]] = []

        for i in tqdm(
            range(0, len(remaining), bs), desc="MobileNet refine", unit="img"
        ):
            batch_paths = remaining[i : i + bs]
            imgs, valids = [], []
            for p in batch_paths:
                try:
                    imgs.append(mv3_tfm(Image.open(p).convert("RGB")))
                    valids.append(p)
                except UnidentifiedImageError:
                    continue

            if not imgs:
                continue

            with torch.no_grad():
                probs = F.softmax(mv3(torch.stack(imgs).to(self.device)), dim=1).cpu()

            for p, prob in zip(valids, probs):
                cat_p = float(prob[cat_ids].max())
                dog_p = float(prob[dog_ids].max())

                if dog_p >= cat_p:
                    dst = root / "dog" / p.name
                    decision = "DOG"
                else:
                    dst = root / "cat" / p.name
                    decision = "CAT"

                self._move(p, dst, mv=move)
                moved += 1
                details.append((p.name, dog_p, cat_p, decision))

                if verbose:
                    print(
                        f"{p.name:30s}  → dog {dog_p:.0%} | cat {cat_p:.0%}  ⇒  {decision}"
                    )

        logging.info("🔍 refine_unknown : %d image(s) déplacées", moved)
        return moved, details

    # ─────────── utilitaires internes ───────────────────────────────────────
    @staticmethod
    def _move(src: Path, dst: Path, mv: bool):
        dst.parent.mkdir(parents=True, exist_ok=True)
        (shutil.move if mv else shutil.copy2)(src, dst)

    @staticmethod
    def _idx_to_label(idx: int, labels: list[str]) -> Label:
        lab = labels[idx].lower()
        if "cat" in lab:
            return "cat"
        if any(
            w in lab
            for w in (
                "dog",
                "hound",
                "terrier",
                "retriever",
                "shepherd",
                "spaniel",
                "collie",
                "poodle",
                "husky",
                "malamute",
                "beagle",
            )
        ):
            return "dog"
        return "unknown"
