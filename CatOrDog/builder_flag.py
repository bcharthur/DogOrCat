#!/usr/bin/env python
"""
CatOrDog.builder_flag
=====================

Construit l’image *flag.png* (noir / blanc) à partir des positions
enregistrées dans  classifiee/cat/  et  classifiee/dog/.

Usage :
    python -m CatOrDog.builder_flag [-d classifiee] [-o flag.png]
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
from PIL import Image

# ─────────────────────────── helpers ────────────────────────────
_NAME_RE = re.compile(r"(\d+)_(\d+)\.[^.]+$")      # fichier → (row, col)


def _collect(folder: Path, value: int):
    """Récupère (row, col, value) pour chaque fichier du dossier."""
    if not folder.exists():
        return []
    for p in folder.iterdir():
        m = _NAME_RE.match(p.name)
        if m:
            yield int(m.group(1)), int(m.group(2)), value


# ─────────────────────────── fonction principale ────────────────
def build_flag(root: Path = Path("classifiee"),
               out: str | Path = "flag.png") -> Path:
    """
    Génère un drapeau N/B à partir de :

        <root>/cat/    → pixels noirs   (0)
        <root>/dog/    → pixels blancs (255)

    Renvoie le chemin du PNG créé.
    """
    cat_pts = list(_collect(root / "cat", 0))
    dog_pts = list(_collect(root / "dog", 255))

    if not cat_pts and not dog_pts:
        raise FileNotFoundError(
            "Aucun fichier de points trouvé dans "
            f"{root}/cat ou {root}/dog – lancez d’abord la classification."
        )

    # Dimensions de l’image (ligne/colonne maxi +1)
    h = max(r for r, _, _ in (*cat_pts, *dog_pts)) + 1
    w = max(c for _, c, _ in (*cat_pts, *dog_pts)) + 1

    # Image initialisée à blanc puis points tracés
    arr = np.full((h, w), 255, dtype=np.uint8)
    for r, c, v in (*cat_pts, *dog_pts):
        arr[r, c] = v

    Image.fromarray(arr).save(out)
    logging.info("✅ flag généré → %s  (%dx%d)", out, w, h)
    return Path(out)


# ─────────────────────────── exécution directe ─────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m CatOrDog.builder_flag",
        description="Construit le flag (N/B) à partir des dossiers cat/ et dog/."
    )
    parser.add_argument("-d", "--dir", default="classifiee", type=Path,
                        help="répertoire racine contenant cat/ et dog/ (def. classifiee)")
    parser.add_argument("-o", "--out", default="flag.png",
                        help="nom du PNG de sortie (def. flag.png)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="mode silencieux (logs WARNING)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)s: %(message)s")

    try:
        build_flag(args.dir, args.out)
    except FileNotFoundError as e:
        logging.error("%s", e)
