import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def get_pano_id(lat, lng, radius=10, min_year=2022):
    """
    Retrieves the closest panorama ID and metadata for a given latitude and longitude.
    
    Args:
        lat (float): Latitude of the location.
        lng (float): Longitude of the location.
        radius (int): Search radius in meters.
        min_year (int): Minimum year for the panorama.
        
    Returns:
        tuple: A tuple containing panorama metadata or None values if not found.
    """
    # URL of the Google Maps API endpoint used in the script
    # This is an internal API that has been reverse engineered (use at own risk).
    url = f"https://maps.googleapis.com/$rpc/google.internal.maps.mapsjs.v1.MapsJsInternalService/SingleImageSearch"

    # Payload structure
    payload = [
        ["apiv3"],
        [[None, None, lat, lng], radius],
        [None, ["en", "US"], None, None, None, None, None, None, [2], None, [[[2, 1, 2], [3, 1, 2], [10, 1, 2]]]],
        [[1, 2, 3, 4, 8, 6]]
    ]

    # Convert payload to JSON
    payload_json = json.dumps(payload)

    headers = {
        "Content-Type": "application/json+protobuf",
        "x-user-agent": "grpc-web-javascript/0.1"
    }

    try:
        # Send POST request to the API
        response = requests.post(url, headers=headers, data=payload_json, timeout=10)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            try:
                # Extract panoId, year, and month from the response
                # This structure is solely based on reverse engineering the internal API.
                pano_id = data[1][5][0][3][0][0][0][1]
                year = data[1][6][7][0]
                month = data[1][6][7][1]
                pano_lat = data[1][5][0][1][0][2]
                pano_lng = data[1][5][0][1][0][3]
                pano_heading = data[1][5][0][6][0][1][3]
                pano_back = data[1][5][0][12][0][1][0]
                pano_front = data[1][5][0][12][0][1][1]
                pano_tilt = data[1][5][0][1][2][1]
                pano_roll = data[1][5][0][1][2][2]
                pano_width = data[1][2][2][1]
                pano_tile_size = data[1][2][3][1][0]
                img_heading = data[1][5][0][1][2][0]
                
                if year < min_year:
                    return (None,) * 13
                    
                return pano_id, year, month, pano_lat, pano_lng, pano_heading, pano_back, pano_front, pano_tilt, pano_roll, pano_width, pano_tile_size, img_heading
            except (IndexError, KeyError, TypeError):
                return (None,) * 13
        else:
            return (None,) * 13
    except requests.exceptions.RequestException:
        return (None,) * 13

def get_pano_ids(coords_list, radius=10, min_year=2022, max_workers=100):
    """
    Takes a list of tuples containing at least latitude and longitude as the first two elements,
    and returns a list of tuples with the panoId, year, and month added at the end.
    Processes the coordinates concurrently to speed up execution and displays progress with tqdm.
    
    Args:
        coords_list (list): List of tuples (lat, lng, ...).
        radius (int): Search radius in meters.
        min_year (int): Minimum year for the panorama.
        max_workers (int): Maximum number of concurrent threads.
        
    Returns:
        list: List of tuples with appended panorama metadata.
    """
    pano_id_set = set()
    updated_coords_list = []

    def process_coord(coord):
        lat, lng = coord[:2]
        result = get_pano_id(lat, lng, radius, min_year)
        return (coord, *result)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor
        futures = {executor.submit(process_coord, coord): coord for coord in coords_list}

        # Process the futures in parallel
        # tqdm to show progress
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing coordinates"):
            try:
                result = future.result()
                coord = result[0]
                pano_data = result[1:]
                pano_id = pano_data[0]
                
                if pano_id and pano_id not in pano_id_set:
                    pano_id_set.add(pano_id)
                    updated_coords_list.append((*coord, *pano_data))
            except Exception as e:
                print(f"Error processing coordinate: {e}")

    return updated_coords_list

