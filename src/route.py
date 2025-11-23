import requests
import polyline
import geopy.distance
import csv
import math
import json

def save_coordinates_to_csv(coordinates: list, filename: str, include_header: bool = False):
    """
    Saves a list of coordinate tuples to a CSV file with an optional header.
    
    Args:
        coordinates (list): A list of tuples, where each tuple can contain additional elements
                            in the following order: (latitude, longitude, heading, panoId, year, month).
        filename (str): The name of the CSV file to save the coordinates.
        include_header (bool): Whether to include a header in the CSV file.
    """
    headers = ['Latitude', 'Longitude', 'Heading', 'point_index', 'PanoId', 'Year', 'Month', 'PanoLat', 'PanoLng', 'PanoHeading', 'PanoBack', 'PanoFront', 'PanoTilt', 'PanoRoll', 'PanoWidth', 'PanoTileSize', 'PanoImageHeading', 'FinalHeading']

    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        if not coordinates:
            print(f"No coordinates to save to {filename}.")
            return

        # Determine the maximum number of elements in the tuples
        max_elements = max(len(coord) for coord in coordinates)

        # Adjust headers based on the maximum tuple length
        final_headers = headers[:max_elements]

        # Write header
        if include_header:
            csvwriter.writerow(final_headers)

        # Write the coordinates
        for coord in coordinates:
            csvwriter.writerow(coord)

    print(f"Coordinates saved to {filename} with{'out' if not include_header else ''} a header.")


def get_detailed_route_with_steps(origin: str, destination: str, api_key: str, waypoints: list = None) -> list:
    """
    Retrieves detailed route using steps from the Google Directions API.
    
    Args:
        origin (str): The starting location.
        destination (str): The destination location.
        api_key (str): Your Google API key.
        waypoints (list, optional): A list of waypoint locations (lat,lng or address).
        
    Returns:
        list: A list of detailed polyline segments (decoded coordinates for all steps).
    """
    url = f'https://maps.googleapis.com/maps/api/directions/json?origin={origin}&destination={destination}&mode=driving&key={api_key}'
    
    if waypoints:
        waypoints_str = '|'.join(waypoints)
        url += f'&waypoints={waypoints_str}'
    
    response = requests.get(url)
    directions_data = response.json()
    
    if 'routes' not in directions_data or not directions_data['routes']:
        raise ValueError("No routes found for the specified locations.")
    
    # Extract the detailed polylines from steps
    detailed_polylines = []
    for leg in directions_data['routes'][0]['legs']:  # Iterate through legs
        for step in leg['steps']:
            encoded_polyline = step['polyline']['points']
            detailed_polylines.append(polyline.decode(encoded_polyline))
    
    return detailed_polylines


def interpolate_route_with_heading(detailed_polylines: list, spacing: float, min_distance: float) -> list:
    """
    Interpolates a route with a given spacing, removes points that are within
    min_distance of the previous point, and calculates the heading for each point.

    Args:
        detailed_polylines (list): A list of lists, where each sublist contains decoded points for a step.
        spacing (float): Desired spacing between points in meters.
        min_distance (float): Minimum distance in meters between points after pruning.

    Returns:
        list: A list of tuples (latitude, longitude, heading, index) for the interpolated and pruned route.
    """
    all_points = []

    # Flatten all polyline segments into a single list
    for polyline_segment in detailed_polylines:
        all_points.extend(polyline_segment)

    # Remove all duplicates from the points
    all_points = remove_all_duplicates(all_points)

    interpolated_points_with_heading = []
    point_index = 0  # Initialize point index

    for i in range(len(all_points) - 1):
        start = all_points[i]
        end = all_points[i + 1]
        distance = geopy.distance.distance(start, end).meters

        if distance > spacing:
            steps = int(distance // spacing)
            for j in range(1, steps + 1):  
                fraction = j / (steps + 1)  
                interpolated_point = (
                    start[0] + (end[0] - start[0]) * fraction,
                    start[1] + (end[1] - start[1]) * fraction
                )
                if interpolated_points_with_heading:
                    # Calculate heading from the last point to the current point
                    heading = calculate_heading(interpolated_points_with_heading[-1][:2], interpolated_point)
                else:
                    heading = 0  # Default heading for the first point
                interpolated_points_with_heading.append((interpolated_point[0], interpolated_point[1], heading, point_index))
                point_index += 1
        # Add the endpoint with its heading
        if interpolated_points_with_heading:
            heading = calculate_heading(interpolated_points_with_heading[-1][:2], end)
        else:
            heading = 0  # Default heading for the first point
        interpolated_points_with_heading.append((end[0], end[1], heading, point_index))
        point_index += 1

    # Prune points that are within min_distance of the previous point
    interpolated_points_with_heading = prune_close_points(interpolated_points_with_heading, min_distance)

    return interpolated_points_with_heading



def calculate_heading(point1, point2):
    """
    Calculates the heading (bearing) between two geographic points.
    
    Args:
        point1 (tuple): Latitude and longitude of the first point (lat1, lon1).
        point2 (tuple): Latitude and longitude of the second point (lat2, lon2).
        
    Returns:
        float: Heading in degrees, measured clockwise from north.
    """
    lat1, lon1 = math.radians(point1[0]), math.radians(point1[1])
    lat2, lon2 = math.radians(point2[0]), math.radians(point2[1])
    
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    
    # Calculate the initial bearing
    initial_bearing = math.atan2(x, y)
    
    # Convert bearing from radians to degrees and normalize to 0-360
    heading = (math.degrees(initial_bearing) + 360) % 360
    return heading

def remove_all_duplicates(points: list, precision: int = 6) -> list:
    """
    Removes all duplicate points (not just consecutive) from a list of coordinates.
    
    Args:
        points (list): A list of tuples (latitude, longitude).
        precision (int): The decimal precision to use when comparing points.
        
    Returns:
        list: A list of unique tuples (latitude, longitude).
    """
    seen = set()
    unique_points = []
    
    for point in points:
        # Round coordinates to the specified precision for comparison
        rounded_point = (round(point[0], precision), round(point[1], precision))
        if rounded_point not in seen:
            seen.add(rounded_point)
            unique_points.append(rounded_point)
    
    return unique_points

def prune_close_points(points: list, min_distance: float) -> list:
    """
    Removes points that are within min_distance meters of the previous point and recalculates headings.

    Args:
        points (list): A list of tuples (latitude, longitude, heading, index).
        min_distance (float): The minimum distance in meters between points.

    Returns:
        list: A list of tuples with points at least min_distance apart.
    """
    if not points:
        return []

    pruned_points = [points[0]]  # Start with the first point

    for point in points[1:]:
        last_point = pruned_points[-1]
        # Extract latitude and longitude for distance calculation
        last_coords = last_point[:2]
        current_coords = point[:2]
        distance = geopy.distance.distance(last_coords, current_coords).meters
        if distance >= min_distance:
            # Recalculate heading from last kept point to current point
            heading = calculate_heading(last_coords, current_coords)
            index = point[3]
            pruned_points.append((current_coords[0], current_coords[1], heading, index))
        else:
            # Optionally, you can skip recalculating the heading
            pass

    return pruned_points


def determine_final_heading(tuples_list):
    """
    Determines the closest heading to new_heading from PanoHeading or (PanoHeading + 180) % 360,
    and appends the closest heading as final_heading to each tuple. The final list is sorted by t[3] (index) (ascending).
    
    Parameters:
        tuples_list (list): A list of tuples in the format 
                            (latitude, longitude, heading, index, pano_id, year, month, PanoLat, PanoLng, PanoBack, PanoHeading, PanoFront, PanoTilt, PanoRoll, PanoWidth, PanoTileSize, PanoImageHeading, final_heading)
    
    Returns:
        list: A new list of tuples, each appended with final_heading and sorted by t[3] (index).
    """
    def closest_heading(pano_heading, new_heading):
        # Calculate the alternative heading (pano_heading + 180 degrees wrapped to 0-360)
        alt_heading = (pano_heading + 180) % 360

        # Function to compute the shortest modular distance between two angles
        def angular_distance(a, b):
            diff = (a - b + 180) % 360 - 180
            return diff if diff != -180 else 180

        # Compute the angular distances
        distance_to_pano = abs(angular_distance(new_heading, pano_heading))
        distance_to_alt = abs(angular_distance(new_heading, alt_heading))

        # Return the heading with the smallest distance
        return pano_heading if distance_to_pano <= distance_to_alt else alt_heading

    result = []
    for t in tuples_list:
        # Extract PanoHeading and new_heading
        pano_heading = t[16] 
        new_heading = t[2]
        
        # Calculate the closest heading
        final_heading = closest_heading(pano_heading, new_heading)
        
        # Append final_heading to the tuple
        result.append(t + (final_heading,))
    
    # Sort the result by t[3] (index) in ascending order
    result.sort(key=lambda x: x[3])
    
    return result


def save_coordinates_to_json(tuples_list, filename):
    """
    Saves the coordinates to a JSON file in a format that can be used on the website map-making.app (in order to visualize the route for debugging).
    
    Args:
        tuples_list (list): List of tuples containing panorama data.
        filename (str): The name of the JSON file to save.
    """
    result = {
        "name": "1",
        "customCoordinates": [],
        "extra": {
            "tags": {},
            "infoCoordinates": []
        }
    }

    for t in tuples_list:
        (
            latitude, longitude, heading, index, pano_id, year, month,
            PanoLat, PanoLng, PanoBack, PanoHeading, PanoFront, PanoTilt, PanoRoll, PanoWidth, PanoTileSize, PanoImageHeading, final_heading
        ) = t  # Unpack the tuple

        coordinate = {
            "lat": PanoLat,
            "lng": PanoLng,
            "heading": final_heading,
            "pitch": 0,
            "zoom": 0,
            "panoId": pano_id,
            "countryCode": None,
            "stateCode": None,
            "extra": {
                "tags": [str(year)],
                "panoId": pano_id,
                "panoDate": f"{year}-{int(month):02d}"
            }
        }

        result["customCoordinates"].append(coordinate)

    with open(filename, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"JSON data has been saved to {filename}")

def process_pano_data(list_of_pano_tuples):
    """
    Processes a list of pano data tuples into a dictionary and an index list.

    Args:
        list_of_pano_tuples (list): List of tuples with pano metadata.

    Returns:
        tuple: A dictionary with pano_id as keys and metadata as values, 
               and a list of indices.
    """
    pano_dict = {}
    index_list = []

    for item in list_of_pano_tuples:
        (
            latitude, longitude, heading, index, pano_id, year, month,
            PanoLat, PanoLng, PanoHeading, PanoBack, PanoFront, PanoTilt, PanoRoll, PanoWidth, PanoTileSize, PanoImageHeading, final_heading
        ) = item  # Unpack the tuple

        # Add the index to the index_list
        index_list.append(index)

        # Store metadata in the dictionary using pano_id as the key
        pano_dict[pano_id] = {
            "latitude": PanoLat,
            "longitude": PanoLng,
            "index": index,
            "year": year,
            "month": month,
            "heading_pano": PanoHeading,
            "heading_back": PanoBack,
            "heading_front": PanoFront,
            "tilt": PanoTilt,
            "roll": PanoRoll,
            "heading_wanted": final_heading,
            "width": PanoWidth,
            "tile_size": PanoTileSize,
            "heading_image": PanoImageHeading
        }

    return pano_dict, index_list
