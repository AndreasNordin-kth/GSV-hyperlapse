import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # API Keys
    GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')
    if not GOOGLE_MAPS_API_KEY:
        raise ValueError("GOOGLE_MAPS_API_KEY not found in environment variables.")

    # Route Parameters
    # Each route is a dictionary with 'origin', 'destination', 'name', and optional 'waypoints' (list of strings)
    ROUTES = [
        {
            "origin": "64.10116707464975, -21.757239877328683",
            "waypoints": ["64.90645071500137, -13.962992053092679", "64.73920074997267, -21.584716006343804"],
            "destination": "64.10117205305254, -21.75726930966471",
            "name": "IS_ringroad"
        }
    ]
    
    # Scanning Parameters
    SPACING = 5  # meters
    MIN_DISTANCE = 0  # meters
    SCANNING_RADIUS = 20  # meters
    MIN_YEAR = 2022 # Minimum year for panoramas (will not accept panoramas older than this year)

    # View Parameters
    VIEW_WIDTH = 2048
    VIEW_HEIGHT = 1152
    FOV_H = 121.24
    ZOOM_LEVEL = 5
    TILE_INTERPOLATION = "INTER_AREA"
    SPHERE_INTERPOLATION = "INTER_AREA"

    VIEW_PARAMS = {
        'view_width': VIEW_WIDTH,
        'view_height': VIEW_HEIGHT,
        'fov_h': FOV_H,
        'zoom_level': ZOOM_LEVEL,
        'tile_interpolation': TILE_INTERPOLATION,
        'sphere_interpolation': SPHERE_INTERPOLATION
    }

    # Data Paths
    DATA_DIR = "data"
    RAW_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
    VIDEO_DIR = os.path.join(DATA_DIR, "video")
    OUTPUT_DIR = os.path.join(DATA_DIR, "output")

    @classmethod
    def ensure_directories(cls):
        os.makedirs(cls.RAW_DIR, exist_ok=True)
        os.makedirs(cls.PROCESSED_DIR, exist_ok=True)
        os.makedirs(cls.VIDEO_DIR, exist_ok=True)
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)

    # Video Parameters
    VIDEO_FPS = 24
    VIDEO_CRF = 20
    VIDEO_PRESET = "medium"
