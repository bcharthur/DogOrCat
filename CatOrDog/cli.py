#!/usr/bin/env python
"""
CatOrDog/cli.py – Interface ligne-de-commande « classique »
(utilisable sans le menu interactif de main.py).

Sous-commandes
──────────────
  classify   → trie a_classer/ → classifiee/ (+ passes sur unknow/)
  clear      → vide classifiee/
  flag       → génère flag.png à partir de classifiee/{cat,dog}
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path

from .classifier import CatDogClassifier
from .builder_flag import build_flag
from .clear import clear_classifiee


# ───────────────────────────── construction du parser
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="catdog-cli",
                                description="Tri automatique chat/chien")
    p.add_argument("--models", nargs="+",
                   default=["resnet50", "efficientnet_b0", "convnext_tiny"],
                   metavar="MODEL",
                   help="ordre des modèles à essayer")
    p.add_argument("--batch", type=int, default=32,
                   help="taille des lots d'inférence")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="mode silencieux (log ERROR)")

    sub = p.add_subparsers(dest="cmd", required=True)

    # classify
    c1 = sub.add_parser("classify", help="tri + passes sur unknow/")
    c1.add_argument("-s", "--src", default="a_classifier", type=Path)
    c1.add_argument("-d", "--dst", default="classifiee", type=Path)
    c1.add_argument("--copy", action="store_true",
                    help="copier au lieu de déplacer le dossier source")
    c1.add_argument("--max-passes", type=int, default=10,
                    help="nombre maxi de reclassements successifs")

    # clear
    c2 = sub.add_parser("clear", help="vide classifiee/")
    c2.add_argument("-d", "--dir", default="classifiee", type=Path)
    c2.add_argument("--rm-dirs", action="store_true",
                    help="supprimer aussi les sous-dossiers vides")

    # flag
    c3 = sub.add_parser("flag", help="génère flag.png depuis classifiee/")
    c3.add_argument("-d", "--dir", default="classifiee", type=Path)
    c3.add_argument("-o", "--out", default="flag.png",
                    help="nom du PNG de sortie (def. flag.png)")

    return p


# ───────────────────────────── point d’entrée CLI
def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.ERROR if args.quiet else logging.INFO,
                        format="%(levelname)s: %(message)s")

    if args.cmd == "clear":
        clear_classifiee(args.dir, remove_dirs=args.rm_dirs)
        return

    if args.cmd == "flag":
        build_flag(args.dir, args.out)
        return

    # cmd == classify
    clf = CatDogClassifier(models_order=args.models,
                           batch_size=args.batch)

    clf.classify_loop(
        src=args.src,
        dst=args.dst,
        copy_src=not args.copy,   # copy_src=True par défaut
        max_passes=args.max_passes
    )


if __name__ == "__main__":
    main()
