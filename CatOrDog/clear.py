from pathlib import Path
from .classifier import CatDogClassifier

def clear_classifiee(root: Path = Path("classifiee"), *,
                     remove_dirs: bool = False) -> None:
    """
    Vide root/{cat,dog,unknow}.
    Si remove_dirs=True, supprime aussi les dossiers vides.
    """
    CatDogClassifier.clear(root, remove_dirs=remove_dirs)
