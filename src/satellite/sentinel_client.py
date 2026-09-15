from sentinelhub import SHConfig, BBox, CRS, DataCollection, SentinelHubRequest, MimeType, bbox_to_dimensions
from typing import List, Dict, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SentinelClient:
    def __init__(self, client_id: str, client_secret: str):
        self.config = SHConfig()
        self.config.sh_client_id = client_id
        self.config.sh_client_secret = client_secret

    def get_ndvi(self, bbox_coords: List[float], time_interval: str) -> np.ndarray:
        """Get NDVI image for given bbox and time."""
        bbox = BBox(bbox= bbox_coords, crs= CRS.WGS84)
        evalscript = """
        //VERSION=3
        function setup() {
            return {
                input: ["B04", "B08"],
                output: { bands: 1 }
            };
        }
        function evaluatePixel(sample) {
            let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
            return [ndvi];
        }
        """
        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=time_interval,
                )
            ],
            responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
            bbox=bbox,
            size=bbox_to_dimensions(bbox, resolution=10),
            config=self.config,
        )
        data = request.get_data()
        return data[0] if data else None

    def get_ndre(self, bbox_coords: List[float], time_interval: str) -> np.ndarray:
        bbox = BBox(bbox=bbox_coords, crs=CRS.WGS84)
        evalscript = """
        //VERSION=3
        function setup() {
            return {
                input: ["B05", "B08A", "B04"],
                output: { bands: 1 }
            };
        }
        function evaluatePixel(sample) {
            let ndre = (sample.B08A - sample.B05) / (sample.B08A + sample.B05);
            return [ndre];
        }
        """
        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=time_interval,
                )
            ],
            responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
            bbox=bbox,
            size=bbox_to_dimensions(bbox, resolution=10),
            config=self.config,
        )
        data = request.get_data()
        return data[0] if data else None