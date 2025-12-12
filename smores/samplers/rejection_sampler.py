"""Probability based item sampler using simple rejection sampling."""

from __future__ import annotations

from csv import DictReader
from pathlib import Path

from .item_sampler import ItemSampler
import smores


class RejectionSampler(ItemSampler):
    """Samples items without replacement according to popularity weights."""

    def __init__(self) -> None:
        super().__init__()
        self._file_path: Path | None = None

    def load_from_file(self, path: Path | str) -> None:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Sampler file not found: {file_path}")

        self._file_path = file_path
        items: list[int] = []
        weights: list[float] = []
        invalid = 0

        with file_path.open(newline="") as handle:
            reader = DictReader(handle)
            if 'item_id' not in reader.fieldnames:
                raise ValueError("Sampler file must contain an 'item_id' column")

            weight_field = 'popularity' if 'popularity' in reader.fieldnames else 'weight'

            for row in reader:
                try:
                    item_id = int(row['item_id'])
                    weight = float(row.get(weight_field, 0.0))
                except (ValueError, TypeError):
                    invalid += 1
                    continue

                if weight <= 0:
                    continue

                if not smores.Smores.state.items.exists_item(item_id):
                    invalid += 1
                    continue

                items.append(item_id)
                weights.append(weight)

        if not items:
            raise ValueError(f"No valid items loaded from {file_path}")

        self.item_ids = items
        self.base_probabilities = weights
        self._normalize_probabilities()

        smores.Smores.state.logger.info(
            f"RejectionSampler loaded {len(self.item_ids)} items from {file_path}"
        )
        if invalid:
            smores.Smores.state.logger.debug(
                f"RejectionSampler skipped {invalid} invalid or unknown items from {file_path}"
            )
