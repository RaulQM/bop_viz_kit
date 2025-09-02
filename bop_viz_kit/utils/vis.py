import trimesh
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from .pyrender import render_offscreen
from scipy.spatial.transform import Rotation as Rot
import pyrender
import os

os.environ['PYOPENGL_PLATFORM'] = 'egl'

# Convert to opengl transformation matrix
def to_opengl_transform(H):
    return H @ trimesh.transformations.rotation_matrix(np.deg2rad(180), [1, 0, 0])

def render_obj(mesh, K, mesh_pose, H, img_width, img_height):
    # create scene config
    ambient_light  = np.array([2.0, 2.0, 2.0])  # [2.5, 2.5, 2.5, 2.5] ; np.array([1.0, 1.0, 1.0, 1.0])
    light_itensity = 0.6
    scene = pyrender.Scene(
        bg_color=np.array([0.0, 0.0, 0.0]), ambient_light=ambient_light
    )
    light = pyrender.SpotLight(
        color = np.ones(3),
        intensity = light_itensity,
        innerConeAngle = np.pi / 16.0,
        outerConeAngle = np.pi / 6.0,
    )
    scene.add(light, pose = H)
    camera = pyrender.IntrinsicsCamera(K[0, 0], K[1, 1], K[0, -1], K[1, -1], znear = 0.05, zfar = 100000)
    # scene.add(mesh, pose = mesh_pose, name = "cad")
    scene.add(pyrender.Mesh.from_trimesh(mesh), pose = mesh_pose, name = "cad")
    scene.add(camera, pose = H)
    r = pyrender.OffscreenRenderer(img_width, img_height)
    color, depth = r.render(scene) #  pyrender.constants.RenderFlags.RGBA flags=pyrender.constants.RenderFlags.FLAT
    return color, depth 

def draw_bounding_box(cvImg, obj_openCV_pose, bbox, intrinsic, color, thickness):
    R, T = obj_openCV_pose[:3, :3], obj_openCV_pose[:3, 3]
    rep = np.matmul(intrinsic, np.matmul(R, bbox.T) + T.reshape(3, 1))
    x = np.int32(rep[0] / rep[2] + 0.5)  # as matplot flip  x axis
    y = np.int32(rep[1] / rep[2] + 0.5)
    bbox_lines = [
        0,
        1,
        0,
        2,
        0,
        4,
        5,
        1,
        5,
        4,
        6,
        2,
        6,
        4,
        3,
        2,
        3,
        1,
        7,
        3,
        7,
        5,
        7,
        6,
    ]
    bbox_lines = [
        0,
        1,
        0,
        2,
        0,
        3,
        1,
        3,
        1,
        2,
        2,
        3,  # face front of
        4,
        5,
        5,
        6,
        6,
        7,
        7,
        4,  # face behind
        0,
        4,
        3,
        7,  # face left
        1,
        5,
        2,
        6,
    ]
    for i in range(len(bbox_lines) // 2):
        id1 = bbox_lines[2 * i]
        id2 = bbox_lines[2 * i + 1]
        cvImg = cv2.line(
            cvImg,
            (x[id1], y[id1]),
            (x[id2], y[id2]),
            color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
    return cvImg


def draw_pose_axis_gt(cvImg, obj_openCV_pose, length_axis, intrinsic, thickness):
    R, T = obj_openCV_pose[:3, :3], np.asarray(obj_openCV_pose[:3, 3]).reshape(3, -1)
    aPts = np.array(
        [[0, 0, 0], [0, 0, length_axis], [0, length_axis, 0], [length_axis, 0, 0]]
    )
    rep = np.matmul(intrinsic, np.matmul(R, aPts.T) + T)
    x = np.int32(rep[0] / rep[2] + 0.5)
    y = np.int32(rep[1] / rep[2] + 0.5)
    cvImg = cv2.line(
        cvImg,
        (x[0], y[0]),
        (x[1], y[1]),
        (0, 0, 255),
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    cvImg = cv2.line(
        cvImg,
        (x[0], y[0]),
        (x[2], y[2]),
        (0, 255, 0),
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    cvImg = cv2.line(
        cvImg,
        (x[0], y[0]),
        (x[3], y[3]),
        (255, 0, 0),
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    return cvImg

def draw_pose_axis(cvImg, obj_openCV_pose, length_axis, intrinsic, thickness):
    R, T = obj_openCV_pose[:3, :3], np.asarray(obj_openCV_pose[:3, 3]).reshape(3, -1)
    aPts = np.array(
        [[0, 0, 0], [0, 0, length_axis], [0, length_axis, 0], [length_axis, 0, 0]]
    )
    rep = np.matmul(intrinsic, np.matmul(R, aPts.T) + T)
    x = np.int32(rep[0] / rep[2] + 0.5)
    y = np.int32(rep[1] / rep[2] + 0.5)
    cvImg = cv2.line(
        cvImg,
        (x[0], y[0]),
        (x[1], y[1]),
        (0, 255, 255),
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    cvImg = cv2.line(
        cvImg,
        (x[0], y[0]),
        (x[2], y[2]),
        (255, 0, 255),
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    cvImg = cv2.line(
        cvImg,
        (x[0], y[0]),
        (x[3], y[3]),
        (255, 255, 0),
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    return cvImg

def draw_pose_contour(
    cvImg, mesh, intrinsic, obj_openCV_pose, color, thickness=3, headless=False
):
    # Mesh 6D pose (aligned with world frame)
    x_ang = 0.0 * np.pi/180.0
    y_ang = 0.0 * np.pi/180.0
    z_ang = 0.0 * np.pi/180.0
    T = np.array([0.0, 0.0, 0.0], dtype = np.float32)
    rot = Rot.from_euler('xyz', [x_ang, y_ang, z_ang])
    R = rot.as_matrix()
    mesh_pose = np.hstack((R, np.expand_dims(T, axis=-1)))
    mesh_pose = np.vstack((mesh_pose, [0, 0, 0, 1]))
    
    H = np.linalg.inv(obj_openCV_pose)
    H = to_opengl_transform(H)

    rendered_color, depth = render_obj(mesh, intrinsic, mesh_pose, H, 640, 480)
    
    #rendered_color, depth = render_offscreen(
    #    mesh, obj_openCV_pose, intrinsic, w=640, h=360, headless=headless
    #)
    
    validMap = (depth > 0).astype(np.uint8)
    
    # find contour
    contours, _ = cv2.findContours(
        validMap, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
    )
    
    # cvImg = cv2.drawContours(cvImg, contours, -1, (255, 255, 255), 1, cv2.LINE_AA) # border
    cvImg = cv2.drawContours(cvImg, contours, -1, color, thickness)
    return rendered_color, cvImg


def draw_point_cloud(cvImg, mesh, K, obj_openCV_pose, color, number_points=500):
    R, T = obj_openCV_pose[:3, :3], obj_openCV_pose[:3, 3]
    meshPts = trimesh.sample.sample_surface(mesh, number_points)[0]  # (N,3)

    pts = np.matmul(K, np.matmul(R, meshPts.T) + T.reshape(-1, 1))
    xs = pts[0] / (pts[2] + 1e-8)
    ys = pts[1] / (pts[2] + 1e-8)

    for pIdx in range(len(xs)):
        cvImg = cv2.circle(cvImg, (int(xs[pIdx]), int(ys[pIdx])), 3, color, -1)
    return cvImg
