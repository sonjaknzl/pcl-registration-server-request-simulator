import open3d as o3d
import numpy as np
import requests
import csv
import copy

# Configuration
POINT_CLOUD_FILE = "target_pcl/1.ply"  
SHAPE_FILE = "shape/shape2.obj"
WALLS_FILE = "walls/walls1.obj"
SERVER_URL = "http://127.0.0.1:5000"  
NUM_REQUESTS = 1  
RADIUS = 3000
POINT_SIZE = 50
FOV = 100


def load_point_cloud(file_path):
    """Loads a single point cloud from a file."""
    return o3d.io.read_point_cloud(file_path)

def load_mesh(file_path):
    """Loads a 3D shape mesh from an OBJ file and rotates it."""
    mesh = o3d.io.read_triangle_mesh(file_path)
    if mesh.is_empty():
        raise ValueError(f"Failed to load mesh from {file_path}")
    
    mesh.compute_vertex_normals() 
    
    # (90° around X)
    R = mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))  
    mesh.rotate(R, center=(0, 0, 0))
   
    return mesh

def visualize_shape_and_point_cloud(pcd, shape_mesh):
    """Visualizes the full point cloud and the shape mesh."""

    shape_mesh.paint_uniform_color([1, 0, 0])  # Shape -> red
    pcd.paint_uniform_color([0, 1, 0])  # PCL -> green

    o3d.visualization.draw_geometries([pcd, shape_mesh])

def get_random_point_on_shape(shape_mesh):
    """Samples a random point from the surface of the shape mesh and returns it as an integer point."""
    
    # Convert mesh vertices and triangles to numpy arrays
    vertices = np.asarray(shape_mesh.vertices)
    triangles = np.asarray(shape_mesh.triangles)
    
    # Check if the mesh is valid
    if len(vertices) == 0 or len(triangles) == 0:
        raise ValueError("The mesh has no vertices or triangles.")
    
    # Compute triangle areas
    p0, p1, p2 = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
    cross_product = np.cross(p1 - p0, p2 - p0)
    triangle_areas = 0.5 * np.linalg.norm(cross_product, axis=1)
    
    # Check for degenerate triangles (zero area)
    if np.any(triangle_areas <= 0):
        raise ValueError("The mesh contains degenerate triangles (zero area).")
    
    # Normalize areas to get a probability distribution
    triangle_probs = triangle_areas / triangle_areas.sum()
    
    # Randomly select a triangle based on weighted probability
    random_triangle_idx = np.random.choice(len(triangles), p=triangle_probs)
    t0, t1, t2 = vertices[triangles[random_triangle_idx]]
    
    # Generate random barycentric coordinates
    r1, r2 = np.random.random(2)
    if r1 + r2 > 1:
        r1, r2 = 1 - r1, 1 - r2
    
    # Compute the random point using barycentric coordinates
    random_point = (1 - r1 - r2) * t0 + r1 * t1 + r2 * t2
    
    # Round the point to the nearest integer
    random_point_rounded = np.round(random_point / 10) * 10
    
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=POINT_SIZE) 
    sphere.translate(random_point)
    sphere.paint_uniform_color([0, 0, 1])  # Blue for visibility
    
    return random_point_rounded, sphere

def visualize_shape_and_point_cloud(pcd, shape_mesh):
    """Visualizes the full point cloud and the shape mesh."""
    shape_mesh.paint_uniform_color([1, 0, 0])  # Shape -> red
    pcd.paint_uniform_color([0, 1, 0])  # PCL -> green
    o3d.visualization.draw_geometries([pcd, shape_mesh])

def get_random_viewing_vector():
    """Generates a normalized random viewing vector in 3D space."""
    theta = np.random.uniform(0, 2 * np.pi) 
    phi = np.random.uniform(-np.pi / 6, np.pi / 6)
    
    x = np.cos(theta) * np.cos(phi)
    y = np.sin(theta) * np.cos(phi)
    z = np.sin(phi)
    
    return np.array([x, y, z]) / np.linalg.norm([x, y, z]) 

def visualize_view_vector(random_point, view_vector, sphere, pcd, radius=50, length=500):
    """Draws a thick cylinder and a line to represent the view vector, starting from random_point."""

    end_point = random_point + view_vector * length

    # Create a cylinder aligned with the Z-axis
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
    
    # Compute the rotation needed to align the cylinder with the view vector
    axis = np.array([0, 0, 1])
    rotation_axis = np.cross(axis, view_vector)

    if np.linalg.norm(rotation_axis) > 1e-6:  # Avoid zero rotation issues
        rotation_axis /= np.linalg.norm(rotation_axis)

    angle = np.arccos(np.clip(np.dot(axis, view_vector), -1.0, 1.0))
    # rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)


    cylinder.rotate(rotation_matrix, center=(0, 0, 0))

    # Translate the cylinder so that its base starts at random_point
    # cylinder.translate(random_point - view_vector * (length / 2))
    cylinder.translate(random_point + view_vector * (length / 2))

    cylinder.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([pcd, sphere, shape_mesh, cylinder])

    return cylinder

def ray_intersects_mesh(ray_origin, ray_end, walls_mesh):
    """
    Check if a ray intersects with the walls mesh using Open3D's raycasting.
    """
    scene = o3d.t.geometry.RaycastingScene()
    
    # Convert legacy TriangleMesh to Tensor-based TriangleMesh
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(walls_mesh)
    scene.add_triangles(mesh_t)

    # Compute ray direction
    ray_direction = ray_end - ray_origin
    ray_length = np.linalg.norm(ray_direction)
    ray_direction = ray_direction / ray_length  # Normalize direction

    # Create ray
    ray = o3d.core.Tensor([np.hstack([ray_origin, ray_direction])], dtype=o3d.core.Dtype.Float32)
    intersection = scene.cast_rays(ray)
    
    # Get distance to intersection
    t_hit = intersection['t_hit'].numpy()[0]

    # Check if intersection occurs before ray_end point
    return t_hit < ray_length

def extract_pcd_around_point(pcd, random_point, view_vector, radius, walls_mesh, fov_degrees):
    """
    Extracts a point cloud within a specified field of view (FOV) and radius, considering occlusion.
    """
    if np.linalg.norm(view_vector) == 0:
        raise ValueError("View vector cannot be zero.")

    view_vector = view_vector / np.linalg.norm(view_vector)

    points = np.asarray(pcd.points)
    distances = np.linalg.norm(points - random_point, axis=1)

    # Filter points within the sphere (distance <= radius)
    sphere_mask = distances <= radius
    print("Points within sphere:", np.sum(sphere_mask))

    # Compute directions from the random_point to each point
    point_directions = points - random_point
    norms = np.linalg.norm(point_directions, axis=1, keepdims=True)
    valid_norms = norms > 0  # Avoid division by zero
    unit_directions = np.where(valid_norms, point_directions / norms, 0)


    dot_products = np.sum(unit_directions * view_vector, axis=1)
    fov_threshold = np.cos(np.radians(fov_degrees / 2))  # cos(60°) = 0.5

    # Keep points within FOV
    fov_mask = dot_products >= fov_threshold
    print(f"Points in {fov_degrees}° FOV:", np.sum(fov_mask))

    # Combine masks to get points within FOV and radius
    final_mask = sphere_mask & fov_mask
    extracted_points = points[final_mask]
    print("Final extracted points:", len(extracted_points))

    # Handle occlusion: check if point is visible from random_point
    visible_points = []
    for point in extracted_points:
        ray_origin = random_point
        ray_end = point

        # Check if ray intersects the walls mesh before reaching the point
        if not ray_intersects_mesh(ray_origin, ray_end, walls_mesh):
            visible_points.append(point)

    print("Visible points after occlusion:", len(visible_points))

    extracted_pcd = o3d.geometry.PointCloud()
    if len(visible_points) > 0:
        extracted_pcd.points = o3d.utility.Vector3dVector(visible_points)

    return extracted_pcd

def visualize_extracted_pcd(pcd, extracted_pcd, random_point, view_vector_visualizer, walls_mesh):
    """Visualizes the extracted hemisphere with occlusion handling."""
    
    # Create sphere to represent random_point
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=50) 
    sphere.translate(random_point)
    sphere.paint_uniform_color([1, 0, 0]) 

    # Visualize
    o3d.visualization.draw_geometries([ extracted_pcd, sphere, view_vector_visualizer, walls_mesh])

def apply_random_transformation(extracted_pcd):
    # Compute the centroid of the extracted point cloud.
    points = np.asarray(extracted_pcd.points)
    centroid_extracted = np.mean(points, axis=0)
    
    # Generate a random yaw rotation (about the Y-axis).
    yaw = np.random.uniform(0, 2 * np.pi)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    R_yaw = np.array([[cos_yaw, 0, sin_yaw],
                      [0,       1, 0],
                      [-sin_yaw,0, cos_yaw]])
    
    # Compute the axis-aligned bounding box of the global point cloud (pcd)
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox.color = (0, 0, 1) 
    
    # Choose a random translation point within the bounding box of pcd.
    min_bound = bbox.get_min_bound()  # [min_x, min_y, min_z]
    max_bound = bbox.get_max_bound()  # [max_x, max_y, max_z]
    random_translation = np.random.uniform(low=min_bound, high=max_bound)
    
    # Calculate the translation vector so that the transformed centroid equals the random point.
    translation = random_translation - R_yaw.dot(centroid_extracted)
    
    # Construct the transformation matrix T_sensor.
    T_sensor = np.eye(4)
    T_sensor[:3, :3] = R_yaw
    T_sensor[:3, 3] = translation
    
    # Apply the transformation to the extracted point cloud.
    extracted_pcd_transformed = copy.deepcopy(extracted_pcd)
    extracted_pcd_transformed.transform(T_sensor)

    pcd.paint_uniform_color([0, 1, 0])                    
    extracted_pcd.paint_uniform_color([0, 0, 1])          
    extracted_pcd_transformed.paint_uniform_color([1, 0, 0]) 

    # o3d.visualization.draw_geometries([pcd, extracted_pcd, extracted_pcd_transformed, bbox])
    
    return extracted_pcd_transformed, T_sensor

def compute_transformation_error(T_sensor, server_transformation):
    
    T_sensor_inv = np.linalg.inv(T_sensor)
    
    print(T_sensor_inv)

    # Convert server response to a numpy array if it isn't already
    T_est = np.array(server_transformation)
    
    # Extract the rotation matrices (top-left 3x3) and translation vectors (first 3 elements of the last column)
    R_sensor = T_sensor_inv[:3, :3]
    R_est    = T_est[:3, :3]
    t_sensor = T_sensor_inv[:3, 3]
    t_est    = T_est[:3, 3]
    
    # Compute the relative rotation matrix (difference between rotations)
    R_rel = R_est @ R_sensor.T
    
    # Compute the rotation error using the formula:
    # angle_error = arccos((trace(R_rel) - 1) / 2)
    angle_error_rad = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1.0, 1.0))
    rot_error_deg = np.degrees(angle_error_rad)
    
    # Compute the translation error as the Euclidean norm of the difference between translation vectors.
    trans_error = np.linalg.norm(t_sensor - t_est)
    
    return rot_error_deg, trans_error

def send_to_server(extracted_pcd, algorithm):
    """Sends the extracted point cloud data to the server."""
    points = np.asarray(extracted_pcd.points).tolist() 
    
    payload = {"sourcepoints": points, "targetmodel": 1, "algorithm": algorithm}  

    # Send POST request
    response = requests.post(SERVER_URL, json=payload)

    # Handle response
    if response.status_code == 201:
        transformation_matrix = response.json().get("transformation_matrix")
        return transformation_matrix
    else:
        print(f"Error: Server responded with {response.status_code}")
        return None  

def write_transformation_to_csv(transformation_matrix, filename):
    """Write the transformation matrix to a CSV file."""
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # If the file is empty, write the header
        if file.tell() == 0:
            writer.writerow(["Algorithm", "Transformation Matrix"])
        
        # Write the transformation matrix along with the algorithm ID
        writer.writerow([1, transformation_matrix])  # Change 1 to the actual algorithm ID if needed

def checkMatrixVisually(extracted_pcd, server_transformation):
        # o3d.visualization.draw_geometries([pcd, extracted_pcd])
        
        server_transformed_pcd = copy.deepcopy(extracted_pcd)
        server_transformation = np.array(server_transformation)
        server_transformed_pcd.transform(server_transformation)
        
        o3d.visualization.draw_geometries([pcd, server_transformed_pcd])

def simulate_requests(num_requests, pcd, shape_mesh, walls_mesh):
    """Runs the simulation, selecting a random point and sending it to the server."""
    # visualize_shape_and_point_cloud(pcd, shape_mesh)
    
    # Get a random point on the shape mesh
    random_point, sphere = get_random_point_on_shape(shape_mesh)
    
    # Get random view vector
    view_vector = get_random_viewing_vector()

    # Draw view vector as a line
    # view_vector_visualizer = visualize_view_vector(random_point, view_vector, sphere, pcd)

    # Extract hemisphere around random point
    extracted_pcd = extract_pcd_around_point(pcd, random_point, view_vector, RADIUS, walls_mesh, FOV)
    
    # visualize_extracted_pcd(pcd, extracted_pcd, random_point, view_vector_visualizer, walls_mesh)
    
    # Apply random transformation
    transformed_extracted_pcd, transformation = apply_random_transformation(extracted_pcd)
    
    
    print("Transformation matrix T_sensor:")
    print(transformation)
    
    # Send extracted point cloud to server
    for algorithm in range(0, 3):  # 1, 2, 3
        status_code = send_to_server(transformed_extracted_pcd, 0)
        print(f"Server response for algorithm {0}: {status_code}")
    
        server_transformation = np.array(status_code)
    
        checkMatrixVisually(transformed_extracted_pcd, server_transformation)
    
        rot_error, trans_error = compute_transformation_error(transformation, server_transformation)
    
        print("Rotation error (degrees):", rot_error)
        print("Translation error:", trans_error)
    
    # Write transformation matrix to CSV


pcd = load_point_cloud(POINT_CLOUD_FILE)
shape_mesh = load_mesh(SHAPE_FILE)
walls_mesh = load_mesh(WALLS_FILE)

simulate_requests(NUM_REQUESTS, pcd, shape_mesh, walls_mesh)