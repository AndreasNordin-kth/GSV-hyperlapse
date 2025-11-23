import asyncio
import numpy as np
import cv2
import os
from aiohttp import ClientSession, ClientTimeout

# ------------------------------ #
#         Helper Functions       #
# ------------------------------ #

def angle_difference(angle1, angle2):
    """Calculate the shortest difference between two angles in degrees."""
    diff = (angle1 - angle2 + 180) % 360 - 180
    return diff

def closest_heading(heading_pano, heading_back, heading_front):
    """
    Determines which heading (back or front) is closer to the desired pano heading.
    """
    # Calculate the shortest angular differences
    diff_back = abs(angle_difference(heading_pano, heading_back))
    diff_front = abs(angle_difference(heading_pano, heading_front))
    
    # Determine the closest heading
    if diff_back <= diff_front:
        return heading_back
    else:
        return heading_front




def get_visible_tiles_grid(pano_params, view_params, img_width=16384, sample_density=100):
    """
    Returns the set of (tile_x, tile_y) indices with (0,0) at the NW corner (as specified by the Street View API).
    of all tiles that will be visible in the final view, using a grid sampling approach.
    This is so that we only download the tiles that are visible in the final view and do not waste time downloading tiles that are not visible.
    
    This method is more robust to distortions and wrapping than the bounding box method.
    
    Args:
        pano_params (dict): Dictionary containing panorama parameters.
        view_params (dict): Dictionary containing view parameters.
        img_width (int): Width of the panorama image.
        sample_density (int): Number of samples per row and column.
    
    Returns:
        set: Set of (tile_x, tile_y) indices of visible tiles.
    """
    # Compute panorama dimensions.
    img_height = img_width / 2

    # Extract panorama parameters.
    heading_pano = pano_params.get("heading_wanted", 0)
    heading_image = pano_params.get("heading_image", 0)
    heading_back = pano_params.get("heading_back", 180)
    heading_front = pano_params.get("heading_front", 0)

    heading_pano_adjusted = closest_heading(heading_pano, heading_back, heading_front)
    
    # Yaw calculation 
    yaw = heading_pano_adjusted - heading_image

    tilt = pano_params.get("tilt", 90)
    roll = pano_params.get("roll", 0)

    # Extract view parameters.
    view_width = view_params.get("view_width", 1920)
    view_height = view_params.get("view_height", 1080)
    fov_h = view_params.get("fov_h", 121.3) #121.3 degrees is the default value for when using google maps on a screen with 16:9 aspect ratio.

    # Convert angles to radians.
    yaw_rad = np.deg2rad(yaw)
    pitch = 90 - tilt  # so that tilt=90 => pitch=0 
    pitch_rad = np.deg2rad(pitch)
    roll_rad = -np.deg2rad(roll)  # Note: we invert roll here.
    
    # Build the composite rotation matrix.
    # R = R_z(roll) * R_x(pitch) * R_y(yaw)
    
    cy = np.cos(yaw_rad)
    sy = np.sin(yaw_rad)
    cp = np.cos(pitch_rad)
    sp = np.sin(pitch_rad)
    cr = np.cos(roll_rad)
    sr = np.sin(roll_rad)

    R_y = np.array([
        [ cy,  0, sy],
        [  0,  1,  0],
        [-sy,  0, cy]
    ])

    R_x = np.array([
        [1,   0,   0],
        [0,  cp, -sp],
        [0,  sp,  cp]
    ])

    R_z = np.array([
        [cr, -sr, 0],
        [sr,  cr, 0],
        [ 0,   0, 1]
    ])

    # Combined rotation.
    R = R_z @ R_x @ R_y

    # To map a ray from camera space into the panorama, undo the above rotation:
    R_inv = R.T

    # Compute the camera's focal length in pixels.
    fov_h_rad = np.deg2rad(fov_h)
    f = (view_width / 2) / np.tan(fov_h_rad / 2)

    # Sample a grid of view pixels.
    # We use a coarser grid for efficiency, but dense enough to catch all tiles.
    # sample_density defines how many points along the larger dimension.
    
    # Ensure we hit the edges
    xs = np.linspace(0, view_width - 1, sample_density)
    ys = np.linspace(0, view_height - 1, sample_density)
    xv, yv = np.meshgrid(xs, ys)
    xv = xv.ravel()
    yv = yv.ravel()

    # Convert view pixel coordinates to camera-space ray directions.
    # Camera coordinate system:
    #   - Origin at the center of the view.
    #   - +x to the right.
    #   - +y upward (so we subtract y from height/2).
    #   - +z forward.
    x_cam = xv - (view_width / 2)
    y_cam = (view_height / 2) - yv
    z_cam = np.full_like(x_cam, f)

    rays_cam = np.vstack((x_cam, y_cam, z_cam)).T
    # Normalize
    rays_cam_norm = rays_cam / np.linalg.norm(rays_cam, axis=1, keepdims=True)

    # Transform the rays into the panorama's coordinate system.
    rays_world = (R_inv @ rays_cam_norm.T).T  # shape (N, 3)

    # Convert world (unit) directions to spherical coordinates.
    # longitude theta = arctan2(x, z)  (range: -pi to pi)
    # latitude  phi = arcsin(y)          (range: -pi/2 to pi/2)
    theta = np.arctan2(rays_world[:, 0], rays_world[:, 2])
    phi = np.arcsin(np.clip(rays_world[:, 1], -1.0, 1.0))

    # Map spherical coordinates to equirectangular image coordinates.
    # u = (theta + pi) / (2pi) * width
    # v = (pi/2 - phi) / pi * height
    u = (theta + np.pi) / (2 * np.pi) * img_width
    v = (np.pi/2 - phi) / np.pi * img_height

    # Wrap u so that u==img_width becomes 0.
    u = np.mod(u, img_width)
    # Clamp v
    v = np.clip(v, 0, img_height - 1)

    # Determine the tile indices (each tile is 512x512).
    tile_size = 512
    tile_x = (u // tile_size).astype(int)
    tile_y = (v // tile_size).astype(int)
    
    visible_tiles = set(zip(tile_x, tile_y))
    
    # Add a buffer of 1 tile around each visible tile to account for edge cases (This part can be removed in order to get fewer tiles but there is a small chance that black edges will appear).
    expanded_visible_tiles = set()
    num_tiles_x = int(np.ceil(img_width / tile_size))
    num_tiles_y = int(np.ceil(img_height / tile_size))

    for x, y in visible_tiles:
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = x + dx, y + dy
                # Handle wrapping for x (longitude)
                nx = nx % num_tiles_x
                # Clamp y (latitude) - do not wrap
                if 0 <= ny < num_tiles_y:
                    expanded_visible_tiles.add((nx, ny))

    return expanded_visible_tiles


def generate_spherical_view(pano_img, pano_params, view_params):
    """
    Generate a spherical view from a panoramic image based on metadata and view parameters.
    """

    interpolation_map = {
        "INTER_NEAREST": cv2.INTER_NEAREST,  # 0
        "INTER_LINEAR": cv2.INTER_LINEAR,  # 1
        "INTER_CUBIC": cv2.INTER_CUBIC,  # 2
        "INTER_AREA": cv2.INTER_AREA,  # 3
        "INTER_LANCZOS4": cv2.INTER_LANCZOS4,  # 4
        "INTER_LINEAR_EXACT": cv2.INTER_LINEAR_EXACT,  # cv::INTER_LINEAR_EXACT
        "INTER_NEAREST_EXACT": cv2.INTER_NEAREST_EXACT,  # cv::INTER_NEAREST_EXACT
        "INTER_MAX": cv2.INTER_MAX,  # cv::INTER_MAX
        "WARP_FILL_OUTLIERS": cv2.WARP_FILL_OUTLIERS,  # cv::WARP_FILL_OUTLIERS
        "WARP_INVERSE_MAP": cv2.WARP_INVERSE_MAP,  # cv::WARP_INVERSE_MAP
    }

    interpolation_str = view_params.get("sphere_interpolation", "INTER_AREA")
    interpolation_method = interpolation_map.get(interpolation_str, cv2.INTER_AREA)

    heading_pano = pano_params.get("heading_wanted", 0)
    heading_image = pano_params.get("heading_image", 0)
    heading_back = pano_params.get("heading_back", 180)
    heading_front = pano_params.get("heading_front", 0)

    heading_pano_adjusted = closest_heading(heading_pano, heading_back, heading_front)

    yaw = heading_pano_adjusted - heading_image
    tilt = pano_params.get("tilt", 90.0)
    roll = pano_params.get("roll", 0.0)

    view_width = view_params.get("view_width", 1920)
    view_height = view_params.get("view_height", 1080)
    fov_h = view_params.get("fov_h", 121.3)

    aspect_ratio = view_height / view_width
    fov_v_rad = np.arctan(np.tan(np.deg2rad(fov_h / 2)) * aspect_ratio) * 2
    fov_v = np.rad2deg(fov_v_rad)

    yaw_rad = np.deg2rad(yaw)
    pitch = 90.0 - tilt
    pitch_rad = np.deg2rad(pitch)
    roll_rad = -np.deg2rad(roll)

    # Rotation matrices
    cos_yaw, sin_yaw = np.cos(yaw_rad), np.sin(yaw_rad)
    R_yaw = np.array([
        [cos_yaw, 0, sin_yaw],
        [0, 1, 0],
        [-sin_yaw, 0, cos_yaw]
    ])

    cos_pitch, sin_pitch = np.cos(pitch_rad), np.sin(pitch_rad)
    R_pitch = np.array([
        [1, 0, 0],
        [0, cos_pitch, -sin_pitch],
        [0, sin_pitch, cos_pitch]
    ])

    cos_roll, sin_roll = np.cos(roll_rad), np.sin(roll_rad)
    R_roll = np.array([
        [cos_roll, -sin_roll, 0],
        [sin_roll, cos_roll, 0],
        [0, 0, 1]
    ])

    # Combined rotation matrix: Roll * Pitch * Yaw
    R_total = R_roll @ R_pitch @ R_yaw

    # To map a ray from camera space into the panorama, we undo the above rotation:
    R_inv = R_total.T

    fov_h_rad = np.deg2rad(fov_h)
    fov_v_rad = np.deg2rad(fov_v)  

    x = np.linspace(-np.tan(fov_h_rad / 2), np.tan(fov_h_rad / 2), view_width)
    y = np.linspace(-np.tan(fov_v_rad / 2), np.tan(fov_v_rad / 2), view_height)
    xv, yv = np.meshgrid(x, -y)

    zv = np.ones_like(xv)
    dirs = np.stack((xv, yv, zv), axis=-1)
    norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
    dirs_normalized = dirs / norms

    dirs_rotated = dirs_normalized.reshape(-1, 3).T
    dirs_rotated = R_inv @ dirs_rotated 
    dirs_rotated = dirs_rotated.T.reshape(view_height, view_width, 3)

    # Spherical coordinates
    theta = np.arctan2(dirs_rotated[..., 0], dirs_rotated[..., 2])
    phi = np.arcsin(np.clip(dirs_rotated[..., 1], -1.0, 1.0)) 

    # Spherical coordinates to equirectangular image coordinates.
    # u = (theta + pi) / (2pi) * width
    # v = (pi/2 - phi) / pi * height
    
    pano_height, pano_width, _ = pano_img.shape
    u = (theta + np.pi) / (2 * np.pi) * pano_width
    v = (np.pi/2 - phi) / np.pi * pano_height

    u = u % pano_width
    v = np.clip(v, 0, pano_height - 1)

    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)

    spherical_view = cv2.remap(
        pano_img,
        map_x,
        map_y,
        interpolation=interpolation_method,
        borderMode=cv2.BORDER_WRAP
    )


    return spherical_view

# ------------------------------ #
#        Asynchronous Download   #
# ------------------------------ #

async def async_fetch_tile(session: ClientSession, pano_id: str, pano_metadata: dict, view_params: dict, x: int, y: int, semaphore: asyncio.Semaphore):
    """
    Asynchronously fetch a single tile using aiohttp.
    Dynamically handles tile size mismatch.
    """
    tile_size = pano_metadata.get("tile_size", 512)
    zoom_level = view_params.get("zoom_level", 5)

    # This is an internal API that has been reverse engineered (use at own risk).
    base_url = (
        f"https://streetviewpixels-pa.googleapis.com/v1/tile?"
        f"cb_client=apiv3&panoid={pano_id}&output=tile&zoom={zoom_level}&nbt=1&fover=2"
    )
    tile_url = f"{base_url}&x={x}&y={y}"

    try:
        async with semaphore:
            async with session.get(tile_url) as response:
                if response.status != 200:
                    response.raise_for_status()
                image_data = await response.read()

        # Decode the image in a separate thread to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        image_np = np.frombuffer(image_data, np.uint8)
        tile = await loop.run_in_executor(None, cv2.imdecode, image_np, cv2.IMREAD_COLOR)

        if tile is None:
            raise ValueError(f"OpenCV failed to decode tile ({x}, {y}) for panoId {pano_id}.")

        # Handle resizing for tiles outside rows 5-10
        if not (5 <= y <= 10) and tile.shape[:2] != (tile_size, tile_size):
            tile = await loop.run_in_executor(None, cv2.resize, tile, (tile_size, tile_size), 0, 0, cv2.INTER_AREA)

        return (x, y, tile)

    except Exception as e:
        print(f"Error fetching tile ({x}, {y}): {e}")
        return (x, y, None)


async def assemble_canvas(visible_tiles, tile_size, canvas_shape, tile_queue):
    """
    Assemble tiles into a canvas, dynamically handling size mismatches.
    """
    canvas = np.zeros(canvas_shape, dtype=np.uint8)

    while True:
        tile_data = await tile_queue.get()
        if tile_data is None: 
            break

        x, y, tile = tile_data
        if tile is not None:
            # Determine the actual tile size
            actual_tile_size = tile.shape[0]

            # Compute the insertion point on the canvas
            top_left_x = x * tile_size
            top_left_y = y * tile_size

            # Add the tile, resizing if necessary
            if actual_tile_size != tile_size:
                tile = cv2.resize(tile, (tile_size, tile_size), interpolation=cv2.INTER_AREA)

            canvas[top_left_y:top_left_y + tile_size, top_left_x:top_left_x + tile_size] = tile

    return canvas


async def async_download_visible_tiles_and_generate_spherical_view(
    pano_id: str,
    pano_metadata: dict,
    view_params: dict,
    session: ClientSession,
    output_dir='output',
    max_concurrent_requests: int = 100
):
    """
    Downloads visible tiles, assembles them, generates the spherical view,
    and saves the resulting image.
    """
    tile_size = pano_metadata.get("tile_size", 512) 
    img_width = pano_metadata.get("width", 16384)
    visible_tiles = get_visible_tiles_grid(pano_params=pano_metadata, view_params=view_params, img_width=16384)

    canvas_height = int(img_width / 2)
    canvas_width = img_width
    canvas_shape = (canvas_height, canvas_width, 3)

    tile_queue = asyncio.Queue()
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    download_tasks = [
        asyncio.create_task(
            async_fetch_tile(session, pano_id, pano_metadata, view_params, x, y, semaphore)
        )
        for x, y in visible_tiles
    ]
    process_task = asyncio.create_task(assemble_canvas(visible_tiles, tile_size, canvas_shape, tile_queue))

    for task in asyncio.as_completed(download_tasks):
        result = await task
        await tile_queue.put(result)

    await tile_queue.put(None)
    canvas = await process_task
    
    index = pano_metadata.get("index", 0)

    # Generate spherical view
    spherical_view_image = generate_spherical_view(canvas, pano_metadata, view_params)

    # Save the spherical view image
    output_filepath = os.path.join(output_dir, f"{index}_{pano_id}.jpg")
    cv2.imwrite(output_filepath, spherical_view_image)

    return output_filepath
