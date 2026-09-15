import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Zone:
    id: str
    polygon: List[Tuple[float, float]]
    area: float
    crop: str = "unknown"
    health_score: float = 0.0


class ZoneMapper:
    def __init__(self, grid_size: int = 4, field_width: float = 100, field_height: float = 100):
        self.grid_size = grid_size
        self.field_width = field_width
        self.field_height = field_height
        self.zones: Dict[str, Zone] = {}
        self.create_zones()  # Initialize zones on construction

    def create_zones(self) -> Dict[str, Zone]:
        self.zones = {}
        rows = cols = self.grid_size
        for r in range(rows):
            for c in range(cols):
                zone_id = f"Z{chr(65+r)}{c+1}"
                x1 = c * self.field_width / cols
                y1 = r * self.field_height / rows
                x2 = (c+1) * self.field_width / cols
                y2 = (r+1) * self.field_height / rows
                polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                area = (x2-x1)*(y2-y1)
                self.zones[zone_id] = Zone(id=zone_id, polygon=polygon, area=area)
        return self.zones

    def assign_detections(self, detections: List[Dict], img_w: int, img_h: int) -> Dict[str, List[Dict]]:
        zone_dets = {zid: [] for zid in self.zones}
        for det in detections:
            # Use center of bbox and scale to field coordinates (0-100)
            cx = (det['bbox'][0] + det['bbox'][2]) / 2 / img_w * self.field_width
            cy = (det['bbox'][1] + det['bbox'][3]) / 2 / img_h * self.field_height
            for zid, zone in self.zones.items():
                if self._point_in_polygon(cx, cy, zone.polygon):
                    det_copy = det.copy()
                    det_copy['zone'] = zid
                    zone_dets[zid].append(det_copy)
                    break
        return zone_dets

    def assign_to_zone(self, x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> str:
        """Assign a single detection to a zone"""
        # Scale to field coordinates (0-100)
        cx = (x1 + x2) / 2 / img_w * self.field_width
        cy = (y1 + y2) / 2 / img_h * self.field_height
        
        for zid, zone in self.zones.items():
            if self._point_in_polygon(cx, cy, zone.polygon):
                return zid
        return 'unknown'

    def _point_in_polygon(self, x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
        inside = False
        n = len(polygon)
        p1x, p1y = polygon[0]
        for i in range(1, n+1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside