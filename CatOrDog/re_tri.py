from pathlib import Path
import logging
from CatOrDog.classifier import CatDogClassifier


def refine_unknow(root: Path = Path("classifiee"),
                  batch_size: int | None = None) -> None:
    unk_dir = root / "unknow"
    total = sum(1 for _ in unk_dir.iterdir()) if unk_dir.exists() else 0
    logging.info("unknow/ avant raffinement : %d image(s)", total)

    clf = CatDogClassifier()
    moved, details = clf.refine_unknown(
        root,
        batch_size=batch_size,
        verbose=True
    )

    for name, dog_p, cat_p, decision in details:
        print(f"{name:30s}  → dog {dog_p:.0%} | cat {cat_p:.0%}  ⇒  {decision}")

    logging.info("Déplacées : %d / %d", moved, total)


