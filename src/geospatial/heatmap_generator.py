import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

# Pattern: optional leading letter prefix, then a letter (row A-Z) and a digit (col 1-9)
# e.g. "ZA1", "A1", "B3"
_ZONE_RE = re.compile(r"([A-Za-z])(\d+)$")


def _parse_zone_id(zone_id: str, grid_size: int):
    """Return (row, col) from zone identifiers like ZA1, A1, B3.
    Returns None if the id cannot be parsed or is out of bounds."""
    try:
        m = _ZONE_RE.search(zone_id)
        if not m:
            return None
        row = ord(m.group(1).upper()) - ord("A")
        col = int(m.group(2)) - 1
        if 0 <= row < grid_size and 0 <= col < grid_size:
            return row, col
    except (ValueError, TypeError, IndexError):
        pass
    logger.warning("Cannot parse zone_id '%s' — skipping", zone_id)
    return None


class HeatmapGenerator:
    """Generate heatmaps from zone-based data (e.g., health scores, pest density)."""
    def __init__(self, grid_size: int = 4):
        self.grid_size = grid_size

    def generate_health_heatmap(self, zone_health: Dict[str, float], output_path: str = None) -> np.ndarray:
        """zone_health: dict mapping zone_id to health score (0-1)"""
        # Create 2D array for heatmap
        data = np.zeros((self.grid_size, self.grid_size))
        for zone_id, score in zone_health.items():
            parsed = _parse_zone_id(zone_id, self.grid_size)
            if parsed is None:
                continue
            row, col = parsed
            data[row, col] = score

        # Plot
        plt.figure(figsize=(6, 6))
        sns.heatmap(data, annot=True, cmap='RdYlGn', vmin=0, vmax=1, cbar_kws={'label': 'Health Score'})
        plt.title('Crop Health Heatmap')
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.close()
            return data
        return data