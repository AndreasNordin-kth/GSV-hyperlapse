import asyncio
import os
import aiohttp
from aiohttp import ClientTimeout
from tqdm import tqdm

from config import Config
from src import route
from src import scan
from src import video
from src import panorama

# Loop through pano_ids to download tiles and generate views with a progress bar
async def process_panoramas(pano_ids, pano_dict, view_params, output_dir):
    connector = aiohttp.TCPConnector(limit_per_host=100, force_close=False)
    timeout = ClientTimeout(total=100)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for pano_id in tqdm(pano_ids, desc="Downloading Tiles and Generating Views"):
            pano_metadata = pano_dict[pano_id]

            await panorama.async_download_visible_tiles_and_generate_spherical_view(
                pano_id=pano_id,
                pano_metadata=pano_metadata,
                view_params=view_params,
                session=session,
                output_dir=output_dir
            )

async def process_single_route(route_config):
    origin = route_config['origin']
    destination = route_config['destination']
    route_name = route_config['name']
    waypoints = route_config.get('waypoints')
    
    print(f"Processing route: {route_name}")

    # Get the polyline for the route
    detailed_polylines = route.get_detailed_route_with_steps(origin, destination, Config.GOOGLE_MAPS_API_KEY, waypoints)

    # Interpolate the route with headings
    interpolated_with_headings = route.interpolate_route_with_heading(detailed_polylines, Config.SPACING, Config.MIN_DISTANCE)
    route.save_coordinates_to_csv(interpolated_with_headings, os.path.join(Config.PROCESSED_DIR, f'{route_name}_interpolated.csv'), include_header=False)

    # Get the pano IDs for the interpolated route
    pano_ids = scan.get_pano_ids(interpolated_with_headings, Config.SCANNING_RADIUS, min_year=Config.MIN_YEAR)

    # Determine final headings
    list_of_pano_tuples = route.determine_final_heading(pano_ids)
    print(f"Number of panoramas found: {len(list_of_pano_tuples)}")

    route.save_coordinates_to_csv(list_of_pano_tuples, os.path.join(Config.PROCESSED_DIR, f'{route_name}.csv'), include_header=True)
    route.save_coordinates_to_json(list_of_pano_tuples, os.path.join(Config.RAW_DIR, f'{route_name}.json'))

    # Process the pano data by creating a dictionary and a list of indices
    pano_dict, index_list = route.process_pano_data(list_of_pano_tuples)

    # Retrieve pano_ids again after processing (keys of the dictionary)
    pano_ids = list(pano_dict.keys())
    
    # Process panoramas
    await process_panoramas(pano_ids, pano_dict, Config.VIEW_PARAMS, os.path.join(Config.OUTPUT_DIR, route_name))

    # Create video
    video.create_high_quality_video(
        os.path.join(Config.OUTPUT_DIR, f"{route_name}"), 
        os.path.join(Config.VIDEO_DIR, f"{route_name}.mp4"), 
        fps=Config.VIDEO_FPS, 
        crf=Config.VIDEO_CRF, 
        preset=Config.VIDEO_PRESET
    )

async def main():
    Config.ensure_directories()
    
    for route_config in Config.ROUTES:
        try:
            await process_single_route(route_config)
        except Exception as e:
            print(f"Error processing route {route_config['name']}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
