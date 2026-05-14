"""
Run WildHands prediction on images. Saves raw joint data only (no mesh, no drawing).

Outputs per hand:
    *_joints.json  — 21 joints (3D model-space + 2D image-space) + bones

Joint order is the raw MANO+tips order WildHands produces (matches the
linux-benchmark-hands archive behaviour). NOTE: this differs from the
OpenPose-style order HaMeR/WiLoR write, so the per-index meaning of
joints_3d/joints_2d is:
    0       wrist
    1..3    index  MCP / PIP / DIP
    4..6    middle MCP / PIP / DIP
    7..9    pinky  MCP / PIP / DIP
    10..12  ring   MCP / PIP / DIP
    13..15  thumb  MCP / PIP / DIP
    16..20  index_tip / middle_tip / ring_tip / pinky_tip / thumb_tip
The JOINT_NAMES / BONES constants below are still the OpenPose-style ones used
elsewhere in this repo; we leave them in for now so the JSON schema is uniform
across algorithms. The mismatch with the actual ordering is intentional and
reproduces the archive's MPJPE numbers (~32-36 mm).

Hand detection: no body detector is used. The full frame is passed to ViTPose,
which produces the hand keypoints used to crop hands for WildHands (same approach
as the modified demo.py).

WildHands needs the camera focal length as input (default 1000, fine for the
provided example images).

Usage:
    python joint_prediction/predict.py
"""

import sys
import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _REPO_ROOT)

# WildHands / HaMeR data locations (must be set before importing those packages)
_DOWNLOADS = os.path.join(_REPO_ROOT, 'downloads')
os.environ.setdefault("CACHE_DIR_HAMER", os.path.join(_DOWNLOADS, "_DATA"))
os.environ.setdefault("HAMER_MANO_DIR", os.path.join(_DOWNLOADS, "_DATA", "data"))
os.environ.setdefault("WILDHANDS_MANO_DIR", os.path.join(_DOWNLOADS, "wildhands"))
os.environ.setdefault("INTRX_PATH", os.path.join(_DOWNLOADS, "wildhands", "intrx.pkl"))

from pathlib import Path
import argparse
import json

import cv2
import numpy as np
import torch

from hamer.utils import recursive_to
import wildhands.common.data_utils as data_utils
from wildhands.configs.parser import construct_args
from wildhands.models.wrapper import WildHandsWrapper as Wrapper
from wildhands.datasets.dataset import WildHandsDataset
from vitpose_model import ViTPoseModel

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CKPT = os.path.join(_DOWNLOADS, "wildhands", "wildhands.ckpt")

BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

JOINT_NAMES = [
    'Wrist',
    'Thumb_CMC', 'Thumb_MCP', 'Thumb_IP', 'Thumb_Tip',
    'Index_MCP', 'Index_PIP', 'Index_DIP', 'Index_Tip',
    'Middle_MCP', 'Middle_PIP', 'Middle_DIP', 'Middle_Tip',
    'Ring_MCP', 'Ring_PIP', 'Ring_DIP', 'Ring_Tip',
    'Pinky_MCP', 'Pinky_PIP', 'Pinky_DIP', 'Pinky_Tip',
]


def save_joints_json(joints_3d, joints_2d, filepath):
    """Save 21 joints (3D model-space + 2D image-space) to JSON."""
    data = {
        'num_joints': 21,
        'joint_names': JOINT_NAMES,
        'joints_3d': [],
        'joints_2d': [],
        'bones': [list(b) for b in BONES],
    }
    for i, name in enumerate(JOINT_NAMES):
        data['joints_3d'].append({
            'id': i, 'name': name,
            'x': float(joints_3d[i, 0]),
            'y': float(joints_3d[i, 1]),
            'z': float(joints_3d[i, 2]),
        })
        data['joints_2d'].append({
            'id': i, 'name': name,
            'x': float(joints_2d[i, 0]),
            'y': float(joints_2d[i, 1]),
        })
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_model(checkpoint, device):
    cfg = construct_args()
    model = Wrapper(cfg)
    ckpt = torch.load(checkpoint, map_location='cpu')
    redundant_keys = ['head_o', 'arti_head', 'grasp_classifier']
    ckpt_params = {k: v for k, v in ckpt['state_dict'].items()
                   if not any(rk in k for rk in redundant_keys)}
    model.load_state_dict(ckpt_params)
    model = model.to(device)
    model.eval()
    return model, cfg


def square_pad_image(cv_img_rgb, render_res):
    """Pad to square (centered) and resize to render_res, like the demo."""
    h, w = cv_img_rgb.shape[:2]
    input_res = max(h, w)
    image = data_utils.generate_patch_image_clean(
        cv_img_rgb, [w / 2, h / 2, input_res, input_res], 1.0, 0.0,
        [render_res, render_res], cv2.INTER_CUBIC)[0]
    return image.clip(0, 255), input_res


def build_intrinsics(cv_img_rgb, focal_length, principal, input_res, render_res):
    h, w = cv_img_rgb.shape[:2]
    px, py = principal
    if px == -1 and py == -1:
        px, py = w / 2, h / 2
    scale = render_res / input_res
    px = px - (w - input_res) / 2
    py = py - (h - input_res) / 2
    return np.array([[focal_length * scale, 0, px * scale],
                     [0, focal_length * scale, py * scale],
                     [0, 0, 1]], dtype=np.float64)


def detect_hands(img_cv2_bgr, cpm):
    """Run ViTPose on the full frame; returns (boxes, right) in the padded-image
    coordinate space, or (None, None) if no hands found."""
    h, w = img_cv2_bgr.shape[:2]
    full_bbox = np.array([[0, 0, w, h, 1.0]])
    vitposes_out = cpm.predict_pose(img_cv2_bgr[:, :, ::-1], [full_bbox])

    bboxes, is_right = [], []
    for vitposes in vitposes_out:
        left_hand_keyp = vitposes['keypoints'][-42:-21]
        right_hand_keyp = vitposes['keypoints'][-21:]
        for keyp, side in ((left_hand_keyp, 0), (right_hand_keyp, 1)):
            valid = keyp[:, 2] > 0.5
            if valid.sum() > 3:
                bboxes.append([keyp[valid, 0].min(), keyp[valid, 1].min(),
                               keyp[valid, 0].max(), keyp[valid, 1].max()])
                is_right.append(side)
    if not bboxes:
        return None, None
    return np.stack(bboxes), np.stack(is_right)


def project_to_image(joints_3d_cam, intrx):
    """Pinhole projection of camera-frame 3D joints with intrinsics K."""
    fx, fy = intrx[0, 0], intrx[1, 1]
    cx, cy = intrx[0, 2], intrx[1, 2]
    joints_2d = np.zeros((joints_3d_cam.shape[0], 2), dtype=np.float64)
    in_front = joints_3d_cam[:, 2] > 0
    joints_2d[in_front, 0] = fx * joints_3d_cam[in_front, 0] / joints_3d_cam[in_front, 2] + cx
    joints_2d[in_front, 1] = fy * joints_3d_cam[in_front, 1] / joints_3d_cam[in_front, 2] + cy
    return joints_2d


def main():
    default_img = os.path.join(_SCRIPT_DIR, 'to-predict')
    default_out = os.path.join(_SCRIPT_DIR, 'to-predict', 'output')

    parser = argparse.ArgumentParser(description='WildHands joint prediction (data only)')
    parser.add_argument('--checkpoint', type=str, default=_DEFAULT_CKPT)
    parser.add_argument('--img_folder', type=str, default=default_img)
    parser.add_argument('--out_folder', type=str, default=default_out)
    parser.add_argument('--focal_length', type=float, default=1000.0,
                        help='Camera focal length for the input images')
    parser.add_argument('--principal', nargs='+', type=float, default=[-1, -1],
                        help='Camera principal point (defaults to image center)')
    parser.add_argument('--render_res', type=int, default=840,
                        help='Square resolution images are padded/resized to before inference')
    parser.add_argument('--rescale_factor', type=float, default=1.75)
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png', '*.jpeg'])
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model, cfg = load_model(args.checkpoint, device)
    cpm = ViTPoseModel(device)

    os.makedirs(args.out_folder, exist_ok=True)
    out_folder_abs = os.path.abspath(args.out_folder)

    img_paths = []
    for ext in args.file_type:
        for p in Path(args.img_folder).glob(ext):
            if not os.path.abspath(str(p)).startswith(out_folder_abs):
                img_paths.append(p)
    if not img_paths:
        print(f'No images found in {args.img_folder}')
        return

    for img_path in sorted(img_paths):
        print(f'Processing: {img_path}')
        cv_img_rgb = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        h, w = cv_img_rgb.shape[:2]

        image, input_res = square_pad_image(cv_img_rgb, args.render_res)
        img_cv2 = image.astype(np.uint8)[:, :, ::-1]  # BGR for ViTPose
        intrx = build_intrinsics(cv_img_rgb, args.focal_length, args.principal,
                                 input_res, args.render_res)

        boxes, right = detect_hands(img_cv2, cpm)
        if boxes is None:
            print('  No hands detected, skipping.')
            continue

        scaled_focal_length = args.focal_length * args.render_res / input_res
        dataset = WildHandsDataset(cfg, image, boxes, right,
                                   focal_length=scaled_focal_length,
                                   rescale_factor=args.rescale_factor, intrx=intrx)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        # transform from padded-image (render_res) coords back to original image coords
        scale_back = input_res / args.render_res
        offset_x = (w - input_res) / 2
        offset_y = (h - input_res) / 2

        img_fn, _ = os.path.splitext(os.path.basename(img_path))
        hand_idx = 0
        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            batch_right = batch[1]['right'].cpu().numpy()
            joints3d_r = out['pred.joints3d.r'].cpu().numpy()
            joints3d_l = out['pred.joints3d.l'].cpu().numpy()
            cam_t_r = out['pred.cam_t.r'].cpu().numpy()
            cam_t_l = out['pred.cam_t.l'].cpu().numpy()

            for n in range(batch_right.shape[0]):
                is_right = bool(batch_right[n] > 0.5)
                joints3d = joints3d_r[n] if is_right else joints3d_l[n]
                cam_t = cam_t_r[n] if is_right else cam_t_l[n]

                joints3d_cam = joints3d + cam_t
                joints2d = project_to_image(joints3d_cam, intrx)
                joints2d[:, 0] = joints2d[:, 0] * scale_back + offset_x
                joints2d[:, 1] = joints2d[:, 1] * scale_back + offset_y

                side = 'right' if is_right else 'left'
                json_path = os.path.join(args.out_folder, f'{img_fn}_hand{hand_idx}_{side}_joints.json')
                save_joints_json(joints3d, joints2d, json_path)
                print(f'  Saved joints: {json_path}')
                hand_idx += 1

    print('Done.')


if __name__ == '__main__':
    main()
