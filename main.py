#!/usr/bin/env python
"""
main.py – Portail interactif & CLI complète
------------------------------------------

Menu (aucun argument) :
  [0] Classifier les images
  [1] Vider les dossiers Dog / Cat / Unknow
  [2] Générer flag.png
  [3] Re-tri du dossier unknow
  [q] Quitter
"""

from __future__ import annotations
import argparse, logging, os, sys
from pathlib import Path
from typing import List

import pyfiglet
from CatOrDog import (
    CatDogClassifier,
    clear_classifiee,
    build_flag,
    refine_unknow,
)

# ───────────────────────────── utilitaires UI
def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def ascii_title() -> None:
    clear_screen()
    print("\n")
    print(pyfiglet.figlet_format("Cat  /  Dog", font="slant"))


def wait_and_clear(msg: str = "Appuyez sur Entrée …") -> None:
    input(msg)
    clear_screen()


# ───────────────────────────── actions de menu
def action_classify(src: Path, dst: Path,
                    models: List[str] | None = None,
                    batch: int = 32,
                    copy: bool = False,
                    max_passes: int = 10):
    CatDogClassifier(models_order=models,
                     batch_size=batch).classify_loop(
        src, dst,
        copy_src=not copy,
        max_passes=max_passes
    )
    wait_and_clear("✔️  Classification terminée. [Entrée]")


def action_clear(dst: Path):
    clear_classifiee(dst)
    wait_and_clear("✔️  Dossiers vidés. [Entrée]")


# ─────────── action_flag
def action_flag(dst: Path, out: str = "flag.png"):
    build_flag(dst, out)          # ← pipeline complet
    wait_and_clear("✔️  flag.png généré + lettres colorisées. [Entrée]")


def action_retri(dst: Path,
                 threshold: float = 0.60,
                 batch: int | None = None):
    refine_unknow(dst, threshold=threshold, batch_size=batch)
    wait_and_clear("✔️  Raffinement terminé. [Entrée]")


# ───────────────────────────── boucle interactive
def interactive_loop():
    while True:
        ascii_title()
        print("[0] - Classifier les images")
        print("[1] - Vider les dossiers Dog, Cat et Unknow")
        print("[2] - Générer flag.png")
        print("[3] - Re-tri de unknow")
        print("[q] - Quitter\n")

        choice = input("Votre choix : ").strip().lower()
        if choice == "0":
            action_classify(Path("a_classer"), Path("classifiee"))
        elif choice == "1":
            action_clear(Path("classifiee"))
        elif choice == "2":
            action_flag(Path("classifiee"))
        elif choice == "3":
            refine_unknow(Path("classifiee"))
            wait_and_clear("✔️  Raffinement terminé. [Entrée]")
        elif choice == "q":
            print("À bientôt !")
            break
        else:
            wait_and_clear("Choix invalide. [Entrée]")


# ───────────────────────────── parser CLI avancé
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="catdog",
                                description="Tri chat/chien complet")
    p.add_argument("--models", nargs="+",
                   default=["resnet50", "efficientnet_b0", "convnext_tiny"])
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("-q", "--quiet", action="store_true")

    sub = p.add_subparsers(dest="cmd", required=True)

    # classify
    c1 = sub.add_parser("classify")
    c1.add_argument("-s", "--src", default="a_classer", type=Path)
    c1.add_argument("-d", "--dst", default="classifiee", type=Path)
    c1.add_argument("--copy", action="store_true")
    c1.add_argument("--max-passes", type=int, default=10)

    # clear
    c2 = sub.add_parser("clear")
    c2.add_argument("-d", "--dir", default="classifiee", type=Path)
    c2.add_argument("--rm-dirs", action="store_true")

    # flag
    c3 = sub.add_parser("flag")
    c3.add_argument("-d", "--dir", default="classifiee", type=Path)
    c3.add_argument("-o", "--out", default="flag.png")

    # retri
    c4 = sub.add_parser("retri")
    c4.add_argument("-d", "--dir", default="classifiee", type=Path)
    c4.add_argument("--thr", type=float, default=0.60,
                    help="seuil proba (def 0.60)")
    c4.add_argument("--batch", type=int, default=None,
                    help="batch MobileNet (def = batch global)")
    return p


# ───────────────────────────── entrée principale
def main(argv: List[str] | None = None):
    if argv is None or len(argv) == 0:
        logging.basicConfig(level=logging.INFO,
                            format="%(levelname)s: %(message)s")
        interactive_loop()
        return

    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.ERROR if args.quiet else logging.INFO,
                        format="%(levelname)s: %(message)s")

    if args.cmd == "classify":
        action_classify(args.src, args.dst,
                        models=args.models,
                        batch=args.batch,
                        copy=args.copy,
                        max_passes=args.max_passes)

    elif args.cmd == "clear":
        clear_classifiee(args.dir, remove_dirs=args.rm_dirs)

    elif args.cmd == "flag":
        build_flag(args.dir, args.out)

    elif args.cmd == "retri":
        refine_unknow(args.dir,
                      threshold=args.thr,
                      batch_size=args.batch)


if __name__ == "__main__":
    main(sys.argv[1:])
