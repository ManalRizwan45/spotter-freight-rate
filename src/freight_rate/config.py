"""Configuration as frozen dataclasses.

Held in Python rather than YAML deliberately: at this size a config file would add a
parser and a dependency while giving up type checking and IDE navigation. Externalised
config earns its keep when non-programmers edit it or when many configs are swept,
neither of which applies here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RANDOM_SEED = 0


@dataclass(frozen=True)
class Paths:
    """Every path the pipeline reads or writes.

    Deliberately not parameterised by a `root` field: the fields would not track it
    (each default is computed from the module-level PROJECT_ROOT), so accepting one
    would invite `Paths(root=tmp)` and silently ignore it. Code that needs a different
    file passes an explicit path instead - see loading.load_train.
    """
    train: Path = PROJECT_ROOT / "data" / "train_test.csv"
    validation: Path = PROJECT_ROOT / "data" / "validation.csv"
    december_inputs: Path = PROJECT_ROOT / "data" / "december_chart_inputs.csv"

    # Deliverables. Both live at the repo root because score.py's documented
    # invocation expects them there.
    predictions: Path = PROJECT_ROOT / "validation_predictions.csv"
    december_predictions: Path = PROJECT_ROOT / "december_predictions.csv"

    figures: Path = PROJECT_ROOT / "reports" / "figures"


@dataclass(frozen=True)
class Fold:
    """One forward-chaining fold. Test always begins after train ends."""
    train_start: str
    train_end: str
    test_start: str
    test_end: str

    @property
    def label(self) -> str:
        return f"train->{self.train_end[5:7]}  test {self.test_start[5:7]}-{self.test_end[5:7]}"


# Expanding window with two-month test blocks, mirroring the real Jan-Oct -> Nov-Dec shape.
FORWARD_FOLDS: tuple[Fold, ...] = (
    Fold("2025-01-01", "2025-04-30", "2025-05-01", "2025-06-30"),
    Fold("2025-01-01", "2025-06-30", "2025-07-01", "2025-08-31"),
    Fold("2025-01-01", "2025-08-31", "2025-09-01", "2025-10-31"),
)


@dataclass(frozen=True)
class ModelParams:
    max_iter: int = 400
    learning_rate: float = 0.06
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 40
    l2_regularization: float = 1.0
    early_stopping: bool = False
    random_state: int = RANDOM_SEED

    def as_kwargs(self) -> dict:
        return dict(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            early_stopping=self.early_stopping,
            random_state=self.random_state,
        )


@dataclass(frozen=True)
class DecemberSettings:
    """Settings for reconstructing december_chart_inputs.csv's missing columns."""
    # quote_signal is estimated from comparable loads: same equipment, similar length.
    reference_equipment: str = "Dry Van"
    distance_band: tuple[int, int] = (250, 500)


@dataclass(frozen=True)
class Config:
    paths: Paths = field(default_factory=Paths)
    folds: tuple[Fold, ...] = FORWARD_FOLDS
    model: ModelParams = field(default_factory=ModelParams)
    december: DecemberSettings = field(default_factory=DecemberSettings)
    seed: int = RANDOM_SEED
    random_kfold_splits: int = 3


CONFIG = Config()
