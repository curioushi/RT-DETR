import blenderproc as bproc
import os
import json
import bpy
import glob
import shutil
import imageio
import numpy as np
import blenderproc.python.renderer.RendererUtility as RendererUtility
from scipy.spatial.transform import Rotation as R

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import colorize_segmap, colorize_depth, colorize_normal, mat44_to_xyzquat, mask_to_bbox

class Box:
    _instance_list = [None]
    def clear_instance_list():
        Box._instance_list = [None]
    def instances():
        return Box._instance_list[1:]
    #            /-------------------------                                                              
    #          /- |                     -/|                                                              
    #        /-   |       5           -/  |                                                              
    #      /-     |                 -/    |                                                              
    #     --------|---------1------/      |               z                                              
    #     |       |                |      |               |      x                                       
    #     |       |                |      |               |     -                                        
    #     |   3   |                |  2   |               |    /                                         
    #     |       |    0           |      |               |  -/                                          
    #     |      /------------------------/               | /                                            
    #     |    /-                  |    /-      y---------|/                                             
    #     |  /-          4         |  /-                                                                 
    #     |/-                      |/-                                                                   
    #     --------------------------                                                                     
    def __init__(self, pose=None, size=None, class_id=None):
        self.pose = pose  # 位姿，通常是一个4x4的变换矩阵
        self.size = np.array(size)  # 尺寸，3D向量 [长, 宽, 高]
        self.area = None  # 表面积
        self.bbox = None  # 边界框 [x, y, width, height]
        self.class_id = class_id  # 类别索引
        self.inst_id = len(Box._instance_list)
        Box._instance_list.append(self)
        self.quads = self.create_quads()
    
    def __repr__(self):
        """Return a detailed string representation of the Box object."""
        size_str = f"size=[{self.size[0]:.3f}, {self.size[1]:.3f}, {self.size[2]:.3f}]" if self.size is not None else "size=None"
        xyzquat = mat44_to_xyzquat(self.pose)
        pose_str = f"pose=[{xyzquat[0]:.3f}, {xyzquat[1]:.3f}, {xyzquat[2]:.3f}, {xyzquat[3]:.3f}, {xyzquat[4]:.3f}, {xyzquat[5]:.3f}, {xyzquat[6]:.3f}]" if self.pose is not None else "pose=None"
        area_str = f"area={self.area}" if self.area is not None else "area=None"
        bbox_str = f"bbox={self.bbox}" if self.bbox is not None else "bbox=None"
        class_id_str = f"class_id={self.class_id}" if self.class_id is not None else "class_id=None"
        inst_id_str = f"inst_id={self.inst_id}" if self.inst_id is not None else "inst_id=None"
        quads_str = ',\n    '.join(repr(q) for q in self.quads)
        quads_str = f"quads=[\n    {quads_str}\n  ]"
        return f"Box(\n  {size_str}\n  {pose_str}\n  {area_str}\n  {bbox_str}\n  {class_id_str}\n  {inst_id_str}\n  {quads_str}\n)"
    
    def create_quads(self):
        quads = []
        for i in range(6):
            quad = Quad(self, i)
            quads.append(quad)
        return quads
    
    def create_blender_object(self):
        for i, quad in enumerate(self.quads):
            if self.class_id == 2 and i == 0:
                continue
            obj = bproc.object.create_primitive("PLANE")
            obj.set_name(f"Box_{self.inst_id:04}({i})")
            obj.set_location(quad.position)
            obj.set_rotation_mat(quad.rotation_mat)
            obj.set_scale(np.array(quad.size + [1.0]) / 2.0)
            obj.set_cp("category_id", self.class_id)
            obj.set_cp("inst_id", self.inst_id)
            obj.set_cp("quad_inst_id", quad.inst_id)

class Quad:
    _instance_list = [None]
    def clear_instance_list():
        Quad._instance_list = [None]
    def instances():
        return Quad._instance_list[1:]

    def __init__(self, parent, index):
        self.parent = parent  # 父Box对象
        self.index = index  # 索引
        self.area = None  # 面积
        self.bbox = None  # 边界框 [x, y, width, height]
        self.inst_id = len(Quad._instance_list)
        Quad._instance_list.append(self)
        self.visibility_2d = None  # 2D可见性（可见面积/总面积）
        self.size_support_vector = [None, None, None, None]  # 尺寸支持向量，4维
    
    @property
    def size(self):
        if self.parent is not None:
            box_size = self.parent.size
            if self.index == 0:
                return [box_size[1], box_size[2]]
            elif self.index == 1:
                return [box_size[1], box_size[2]]
            elif self.index == 2:
                return [box_size[2], box_size[0]]
            elif self.index == 3:
                return [box_size[2], box_size[0]]
            elif self.index == 4:
                return [box_size[0], box_size[1]]
            elif self.index == 5:
                return [box_size[0], box_size[1]]
            else:
                raise ValueError(f"Invalid index: {self.index}")
        return None
    
    @property
    def rotation_mat(self):
        if self.parent is not None:
            box_rotation = self.parent.pose[:3, :3]
            x_axis = box_rotation[:, 0]
            y_axis = box_rotation[:, 1]
            z_axis = box_rotation[:, 2]
            mat33 = np.eye(3)
            if self.index == 0:
                mat33[:3, 0] = -y_axis
                mat33[:3, 1] = z_axis
                mat33[:3, 2] = -x_axis
            elif self.index == 1:
                mat33[:3, 0] = y_axis
                mat33[:3, 1] = z_axis
                mat33[:3, 2] = x_axis
            elif self.index == 2:
                mat33[:3, 0] = z_axis
                mat33[:3, 1] = -x_axis
                mat33[:3, 2] = -y_axis
            elif self.index == 3:
                mat33[:3, 0] = z_axis
                mat33[:3, 1] = x_axis
                mat33[:3, 2] = y_axis
            elif self.index == 4:
                mat33[:3, 0] = x_axis
                mat33[:3, 1] = -y_axis
                mat33[:3, 2] = -z_axis
            elif self.index == 5:
                mat33[:3, 0] = x_axis
                mat33[:3, 1] = y_axis
                mat33[:3, 2] = z_axis
            else:
                raise ValueError(f"Invalid index: {self.index}")
            return mat33
        return None
    
    @property
    def position(self):
        if self.parent is not None:
            box_size = self.parent.size
            box_position = self.parent.pose[:3, 3]
            box_rotation = self.parent.pose[:3, :3]
            x_axis = box_rotation[:, 0]
            y_axis = box_rotation[:, 1]
            z_axis = box_rotation[:, 2]
            if self.index == 0:
                return box_position - x_axis * box_size[0] / 2
            elif self.index == 1:
                return box_position + x_axis * box_size[0] / 2
            elif self.index == 2:
                return box_position - y_axis * box_size[1] / 2
            elif self.index == 3:
                return box_position + y_axis * box_size[1] / 2
            elif self.index == 4:
                return box_position - z_axis * box_size[2] / 2
            elif self.index == 5:
                return box_position + z_axis * box_size[2] / 2
            else:
                raise ValueError(f"Invalid index: {self.index}")
        return None
    
    @property
    def pose(self):
        position = self.position
        rotation_mat = self.rotation_mat
        if position is not None and rotation_mat is not None:
            pose = np.eye(4)
            pose[:3, :3] = rotation_mat
            pose[:3, 3] = position
            return pose
        return None
    
    def __repr__(self):
        box_size = self.size
        box_pose = self.pose
        xyzquat = mat44_to_xyzquat(box_pose)
        return  f"Quad(index={self.index}, inst_id={self.inst_id}, size=[{box_size[0]:.3f}, {box_size[1]:.3f}])" \
                f", pose=[{xyzquat[0]:.3f}, {xyzquat[1]:.3f}, {xyzquat[2]:.3f}, {xyzquat[3]:.3f}, {xyzquat[4]:.3f}, {xyzquat[5]:.3f}, {xyzquat[6]:.3f}]"


base_dir = "data/dropped_box_2024_1220_1K/data"
folders = glob.glob(f"{base_dir}/*/")
json_files = sorted([os.path.join(folder, "data.json") for folder in folders])

bproc.init()
bproc.renderer.enable_normals_output()
bproc.renderer.enable_depth_output(activate_antialiasing=False)
for index, json_path in enumerate(json_files):
    print(f"processing {index}/{len(json_files)}")
    with open(json_path, "r") as f:
        json_data = json.load(f)

    output_dir = f"output/{index:04}"
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    class_to_id = {
        "box": 1,
        "container": 2,
    }

    boxes = []
    Box.clear_instance_list()
    Quad.clear_instance_list()
    for box_data in sorted(json_data["boxes"], key=lambda x: x["position"][0])[:200]:
        size = box_data["size"]
        position = box_data["position"]
        quat_xyzw = box_data["rotation"]
        rotation_matrix = R.from_quat(quat_xyzw).as_matrix()
        pose = np.eye(4)
        pose[:3, :3] = rotation_matrix
        pose[:3, 3] = position
        box = Box(pose=pose, size=size, class_id=class_to_id["box"])
        boxes.append(box)
    
    container_data = json_data["container"]
    container_length, container_width, container_height = container_data["size"]
    position = container_data["position"]
    quat_xyzw = container_data["rotation"]
    rotation_matrix = R.from_quat(quat_xyzw).as_matrix()
    x_axis = rotation_matrix[:, 0]
    y_axis = rotation_matrix[:, 1]
    z_axis = rotation_matrix[:, 2]
    position = position - x_axis * container_length / 2 + z_axis * container_height / 2  # move the origin from from-bottom to center
    pose = np.eye(4)
    pose[:3, :3] = rotation_matrix
    pose[:3, 3] = position
    container = Box(pose=pose, size=[container_length, container_width, container_height], class_id=class_to_id["container"])
    boxes.append(container)
    
    for i, box in enumerate(boxes):
        box.create_blender_object()

    image_width = 512
    image_height = 512
    fov_degrees = 120.0
    fov_rad = np.radians(fov_degrees)
    fx = image_width / (2 * np.tan(fov_rad / 2))
    fy = fx
    cx = image_width / 2
    cy = image_height / 2

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    bproc.camera.set_intrinsics_from_K_matrix(K, image_width=image_width, image_height=image_height)
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = rotation_matrix @ R.from_euler('yx', [-90, 90], degrees=True).as_matrix()
    camera_pose[:3, 3] = container_data["position"] \
                        - rotation_matrix[:, 0] * container_length \
                        + rotation_matrix[:, 2] * container_height / 2

    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location(camera_pose[:3, 3])
    light.set_energy(1)

    bproc.camera.add_camera_pose(camera_pose, frame=0)
    bproc.renderer.set_render_devices(desired_gpu_device_type="CUDA")
    bpy.context.scene.cycles.samples = 1
    bpy.context.scene.cycles.use_denoising = False
    bpy.context.scene.cycles.max_bounces = 1
    bproc.renderer.set_world_background(color=[1,1,1], strength=1)
    data = bproc.renderer.render()
    segmap_dict = bproc.renderer.render_segmap(map_by=["class", "instance", "cp_inst_id", "cp_quad_inst_id"], 
                                            default_values={"class": 0, "instance": 0, "cp_inst_id": 0, "cp_quad_inst_id": 0})

    # relocate the camera based on the boxes in current viewport
    depth_image_raw = data["depth"][0]
    class_segmap = segmap_dict["class_segmaps"][0]
    box_mask = class_segmap == class_to_id["box"]
    box_depth = depth_image_raw[box_mask]
    if box_depth.size > 0:
        mean_box_depth = np.percentile(box_depth, 70)
    else:
        mean_box_depth = 1.2

    offset_camera_x = 1.2 - mean_box_depth
    camera_pose[:3, 3] += offset_camera_x * camera_pose[:3, 2]

    # render the whole scene
    image_width = 1024
    image_height = 1024
    fov_degrees = 120.0
    fov_rad = np.radians(fov_degrees)
    fx = image_width / (2 * np.tan(fov_rad / 2))
    fy = fx
    cx = image_width / 2
    cy = image_height / 2

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    bproc.camera.set_intrinsics_from_K_matrix(K, image_width=image_width, image_height=image_height)

    RendererUtility.render_init()
    bpy.context.scene.cycles.samples = 50
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.cycles.max_bounces = 3
    bproc.camera.add_camera_pose(camera_pose, frame=0)
    bproc.renderer.enable_normals_output()
    data = bproc.renderer.render()
    segmap_dict = bproc.renderer.render_segmap(map_by=["class", "instance", "cp_inst_id", "cp_quad_inst_id"], 
                                            default_values={"class": 0, "instance": 0, "cp_inst_id": 0, "cp_quad_inst_id": 0})
    color_image = data["colors"][0]
    normal_image = data["normals"][0]
    depth_image_raw = data["depth"][0]
    normal_image = normal_image / np.linalg.norm(normal_image, axis=-1, keepdims=True)
    classmap = segmap_dict["class_segmaps"][0]
    instmap = segmap_dict["cp_inst_id_segmaps"][0]
    quad_instmap = segmap_dict["cp_quad_inst_id_segmaps"][0]
    colorized_classmap = colorize_segmap(classmap)
    colorized_instmap = colorize_segmap(instmap)
    colorized_quad_instmap = colorize_segmap(quad_instmap)
    colorized_depth_image = colorize_depth(depth_image_raw)
    colorized_normal_image = colorize_normal(normal_image)
    visible_box_ids = np.unique(instmap)
    for box_id in visible_box_ids:
        if box_id == 0:
            continue
        box = Box._instance_list[box_id]
        box_mask = instmap == box_id
        box.area = int(np.sum(box_mask))
        box.bbox = mask_to_bbox(box_mask)
    visible_quad_ids = np.unique(quad_instmap)
    for quad_id in visible_quad_ids:
        if quad_id == 0:
            continue
        quad = Quad._instance_list[quad_id]
        quad_mask = quad_instmap == quad_id
        quad.area = int(np.sum(quad_mask))
        quad.bbox = mask_to_bbox(quad_mask)

    imageio.imwrite(os.path.join(output_dir, "color.png"), color_image)
    np.savez_compressed(os.path.join(output_dir, "depth.npz"), depth_image_raw)
    np.savez_compressed(os.path.join(output_dir, "normal.npz"), normal_image)
    np.savez_compressed(os.path.join(output_dir, "classmap.npz"), classmap)
    np.savez_compressed(os.path.join(output_dir, "instmap.npz"), instmap)
    np.savez_compressed(os.path.join(output_dir, "quad_instmap.npz"), quad_instmap)
    imageio.imwrite(os.path.join(output_dir, "colorized_classmap.png"), colorized_classmap)
    imageio.imwrite(os.path.join(output_dir, "colorized_instmap.png"), colorized_instmap)
    imageio.imwrite(os.path.join(output_dir, "colorized_quad_instmap.png"), colorized_quad_instmap)
    imageio.imwrite(os.path.join(output_dir, "colorized_depth.png"), colorized_depth_image)
    imageio.imwrite(os.path.join(output_dir, "colorized_normal.png"), colorized_normal_image)
    json_data = dict()
    json_data["camera"] = {
        "position": camera_pose[:3, 3].tolist(),
        "rotation": (R.from_matrix(camera_pose[:3, :3]) * R.from_euler('x', 180, degrees=True)).as_quat().tolist(),
        "intrinsics": K.tolist(),
        "image_width": image_width,
        "image_height": image_height,
    }
    json_data["boxes"] = [
        {
            "position": box.pose[:3, 3].tolist(),
            "rotation": R.from_matrix(box.pose[:3, :3]).as_quat().tolist(),
            "size": box.size.tolist(),
            "area": box.area,
            "bbox": box.bbox,
            "instance_id": box.inst_id,
            "category_id": box.class_id,
        } for box in Box.instances() if box.area is not None
    ]
    json_data["quads"] = [
        {
            "position": quad.pose[:3, 3].tolist(),
            "rotation": R.from_matrix(quad.pose[:3, :3]).as_quat().tolist(),
            "size": quad.size,
            "area": quad.area,
            "bbox": quad.bbox,
            "instance_id": quad.inst_id,
            "parent_instance_id": quad.parent.inst_id,
        } for quad in Quad.instances() if quad.area is not None
    ]
    with open(os.path.join(output_dir, "scene.json"), "w") as f:
        json.dump(json_data, f, indent=2)

    bproc.clean_up(clean_up_camera=True)
