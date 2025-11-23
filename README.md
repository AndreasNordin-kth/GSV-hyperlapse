# GSV Hyperlapse

Generate hyperlapse videos from Google Street View panoramas using the Google Maps Directions API and the internal Google Street View API.

## Example

Example videos generated with this program:

[![GSV Hyperlapse Example](https://img.youtube.com/vi/rnyDKRLcgVc/maxresdefault.jpg)](https://www.youtube.com/playlist?list=PL561EFKo4EBD9_QUPP5TOLFGtjh5tVLxi)


## About

This tool allows you to create hyperlapse videos by downloading Google Street View panoramas along a route, created via the Google Maps Directions API. 
The program retrieves the route points and downloads the valid panoramas along the route (in the correct order), then converts the panoramas into a video. The heading is always set to the direction of the route (this means that the camera will sometimes be looking backwards relative to the driving direction) and the pitch is always set to 0.

Note: Downloading panoramas (at 1920x1080 resolution) take around 0.4-0.6 seconds and the images take up around 0.8mb. In practice I have found that the program runs at around 80 km/hour.

## Prerequisites

-   **Python (with dependencies)**
-   **FFmpeg**: Must be installed and accessible in your system's PATH.
-   **Google Maps API Key**: Required for the Directions API (to calculate routes).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AndreasNordin-kth/GSV-Hyperlapse.git
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Environment Variables:**
    Create a `.env` file in the root directory and add your Google Maps API Key (only directions API is used):
    ```env
    GOOGLE_MAPS_API_KEY=your_api_key_here
    ```

## Usage

1.  **Configure the Route:**
    Open `config.py` and modify the `ROUTES` list (there can be multiple routes) with your origin, destination, waypoints (can be empty), and name.
    ```python
    ROUTES = [
        {
            "origin": "Starting point",
            "waypoints": ["Waypoint 1", "Waypoint 2", ...],
            "destination": "Destination point",
            "name": "name"
        }
    ]
    ```

2.  **Run the Script:**
    ```bash
    python main.py
    ```

    The script will:
    -   Calculate the route.
    -   Find available panoramas.
    -   Download tiles and generate frames.
    -   Save the final video in `data/video/`.

## Configuration

You can adjust various settings in `config.py`:

-   **Scanning**: `SPACING` (meters between frames), `MIN_YEAR` (panoramas older than this year will not be used).
-   **View**: `VIEW_WIDTH` (width resolution), `VIEW_HEIGHT` (height resolution), `FOV_H` (field of view).
-   **Video**: `VIDEO_FPS` (frames per second), `VIDEO_CRF` (quality), `VIDEO_PRESET` (encoding speed).

## Disclaimer

This tool uses the internal Google Street View API for streetview imagery and metadata retrieval. Use at your own risk.
