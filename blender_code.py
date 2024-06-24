import bpy
import math
import mathutils
import os
import numpy
import json

def clear_all_cameras():
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            bpy.data.objects.remove(obj, do_unlink=True)

def look_at(obj_camera, target_pos):
    direction = target_pos - obj_camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    obj_camera.rotation_euler = rot_quat.to_euler()

def create_camera(location, focal_length=50, sensor_width=36):
    bpy.ops.object.camera_add(location=location)
    camera = bpy.context.object
    return camera, camera.data.angle_x

def render_scene(output_path, camera):
    bpy.context.scene.camera = camera
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)

def save_pose_matrix(camera):
    matrix = camera.matrix_world
    rows = []
    for row in matrix:
        rows.append(list(row))            
    return rows
    

RESOLUTION = 100
NUM_CAMERAS = 106 # Number of cameras around object in upper hemisphere
RADIUS = 5
CENTER_POINT = mathutils.Vector((0, 0, 0))

bpy.context.scene.render.resolution_x = RESOLUTION
bpy.context.scene.render.resolution_y = RESOLUTION

num_cameras = NUM_CAMERAS

output_dir = "." # Change this to the directory where you want to save the images & pose data
os.makedirs(output_dir, exist_ok=True)

final_pose_data = {}
frames = []

clear_all_cameras()

nc = NUM_CAMERAS * 2 # Doubling because we'll be disregarding all cameras in lower hemisphere
for i in range(nc):
    frame_obj = {}
    
    phi = math.acos(1 - 2 * (i + 0.5) / nc)  # Polar angle
    theta = math.pi * (1 + 5**0.5) * (i + 0.5)  # Azimuthal angle
    
    # Convert spherical coordinates to Cartesian coordinates
    x = RADIUS * math.sin(phi) * math.cos(theta)
    y = RADIUS * math.sin(phi) * math.sin(theta)
    z = RADIUS * math.cos(phi)

    # Z coordinate of less than 0 means the camera is below the object, i.e. the lower hemisphere
    if z < 0:
        continue

    cam_location = mathutils.Vector((x, y, z))
    camera, camera_angle_x = create_camera(cam_location)
    look_at(camera, CENTER_POINT)

    image_name = f'render_{i:03d}.png'
    render_scene(os.path.join(output_dir, image_name), camera)
    frame_obj["file_path"] = image_name
    frame_obj["transform_matrix"] = save_pose_matrix(camera)
    
    frames.append(frame_obj)
        
final_pose_data["camera_angle_x"] = camera_angle_x
final_pose_data["frames"] = frames

with open(os.path.join(output_dir, 'data.json'), "w", encoding='utf-8') as f:
    f.write(json.dumps(final_pose_data, indent=2))
        
