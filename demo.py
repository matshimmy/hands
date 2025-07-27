import os

os.environ["CACHE_DIR_HAMER"] = "downloads/_DATA"
os.environ["HAMER_MANO_DIR"] = "downloads/_DATA/data"
os.environ["WILDHANDS_MANO_DIR"] = "downloads/wildhands"
os.environ["INTRX_PATH"] = "downloads/wildhands/intrx.pkl"

from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
import pickle

from hamer.models import load_hamer
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils.renderer import Renderer, cam_crop_to_full

from wildhands.configs.parser import construct_args
from wildhands.models.wrapper import WildHandsWrapper as Wrapper
from wildhands.datasets.dataset import WildHandsDataset
import wildhands.common.data_utils as data_utils

from vitpose_model import ViTPoseModel

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)


def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--img_folder', type=str, default='downloads/example_data', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='out', help='Output folder to save rendered results')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')
    parser.add_argument('--render_res', type=int, default=840, help='Resolution for rendering')
    parser.add_argument('--focal_length', type=float, default=1000, help='Camera focal length corresponding to the input image')
    parser.add_argument('--principal', nargs='+', default=[-1, -1], help='Camera principal point corresponding to the input image')
    args = parser.parse_args()
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    cfg = construct_args()
    model = Wrapper(cfg)
    ckpt = torch.load("downloads/wildhands/wildhands.ckpt", map_location='cpu')
    ckpt_params = {}
    redundant_keys = ['head_o', 'arti_head', 'grasp_classifier']
    for k, v in ckpt["state_dict"].items():
        if not any([rk in k for rk in redundant_keys]):
            ckpt_params[k] = v
    model.load_state_dict(ckpt_params)
    model = model.to(device)
    model.eval()

    renderer = Renderer(cfg, faces = model.model.mano_r.mano.faces)

    # keypoint detector
    cpm = ViTPoseModel(device)

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    # Get all demo images ends with .jpg or .png
    img_paths = sorted([img for end in args.file_type for img in Path(args.img_folder).glob(end)])

    # Iterate over all images in folder
    for img_path in tqdm(img_paths):

        # square images are convenient since different models have different input sizes
        cv_img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        input_res = max(cv_img.shape[:2])
        image = data_utils.generate_patch_image_clean(cv_img, [cv_img.shape[1]/2, cv_img.shape[0]/2, input_res, input_res], 1.0, 0.0, [args.render_res, args.render_res], cv2.INTER_CUBIC)[0]
        img = image.clip(0, 255)
        img_cv2 = img.astype(np.uint8)[..., ::-1]

        intrx = None

        px, py = args.principal
        if px == -1 and py == -1:
            px = cv_img.shape[1] / 2
            py = cv_img.shape[0] / 2
        intrx = np.array([[args.focal_length, 0, px], [0, args.focal_length, py], [0, 0, 1]])

        # transform intrx as per padded image
        scale = args.render_res / input_res
        px = px - (cv_img.shape[1] - input_res) / 2
        py = py - (cv_img.shape[0] - input_res) / 2
        intrx[0, 2] = px * scale
        intrx[1, 2] = py * scale
        intrx[0, 0] *= scale
        intrx[1, 1] *= scale

        # Process the whole image - create a bounding box for the entire image
        img_height, img_width = img_cv2.shape[:2]
        whole_image_bbox = np.array([[0, 0, img_width, img_height, 1.0]])  # [x1, y1, x2, y2, confidence]

        # Detect human keypoints for the whole image
        vitposes_out = cpm.predict_pose(
            img_cv2,
            [whole_image_bbox],
        )

        bboxes = []
        is_right = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Rejecting not confident detections
            keyp = left_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(0)
            keyp = right_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(1)

        if len(bboxes) == 0:
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        scaled_focal_length = args.focal_length * args.render_res / input_res
        dataset = WildHandsDataset(cfg, img, boxes, right, focal_length=scaled_focal_length, rescale_factor=1.75, intrx=intrx)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        
        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            batch_right = batch[1]['right'] # batch = (inputs, meta_info)
            pred_cam_t_full_r_wh = out['pred.cam_t.r'].cpu().numpy()
            pred_cam_t_full_l_wh = out['pred.cam_t.l'].cpu().numpy()
            pred_vertices_r = out['pred.vertices.r'].cpu().numpy()
            pred_vertices_l = out['pred.vertices.l'].cpu().numpy()
            
            batch_size = batch_right.shape[0]
            for n in range(batch_size):

                # Add all verts and cams to list
                is_right = batch_right[n].cpu().numpy()
                verts = pred_vertices_r[n] if is_right else pred_vertices_l[n]
                cam_t = pred_cam_t_full_r_wh[n] if is_right else pred_cam_t_full_l_wh[n]
                
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)

        # Render hands onto the image
        # if len(all_verts) > 0:
        #     misc_args = dict(mesh_base_color=LIGHT_BLUE, scene_bg_color=(1, 1, 1), focal_length=scaled_focal_length)

        #     input_img = cv2.resize(img_cv2, (args.render_res, args.render_res), interpolation=cv2.INTER_CUBIC)
        #     input_img = input_img.astype(np.float32)[:,:,::-1]/255.0
        #     input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2)

        #     cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=[args.render_res, args.render_res], is_right=all_right, **misc_args)
        #     input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]
            
        #     # Get filename from path img_path
        #     img_fn, _ = os.path.splitext(os.path.basename(img_path))
        #     cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}.jpg'), 255*input_img_overlay[:, :, ::-1])

if __name__ == '__main__':
    main()
