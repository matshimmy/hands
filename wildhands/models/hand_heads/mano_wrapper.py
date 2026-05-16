import torch
from typing import Optional
import smplx
from smplx.lbs import lbs
from smplx.utils import MANOOutput, Tensor


# Permutation from smplx's native MANO output order (wrist, then
# index/middle/pinky/ring/thumb proximal->distal, then 5 tips in
# thumb/index/middle/ring/pinky order) to the canonical 21-joint
# OpenPose-style order (wrist, thumb, index, middle, ring, pinky;
# 4 joints per finger). Matches HaMeR's mano_wrapper.MANO.joint_map.
MANO_TO_OPENPOSE = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]


class MANO(smplx.MANO):
    def __init__(self, *args, **kwargs):
        super(MANO, self).__init__(*args, **kwargs)
        self.register_buffer('joint_map', torch.tensor(MANO_TO_OPENPOSE, dtype=torch.long), persistent=False)

    def forward(
        self,
        betas: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        hand_pose: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        return_verts: bool = True,
        return_full_pose: bool = False,
        **kwargs
    ) -> MANOOutput:
        ''' Forward pass for the MANO model
        '''
        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = (global_orient if global_orient is not None else
                         self.global_orient)
        betas = betas if betas is not None else self.betas
        hand_pose = (hand_pose if hand_pose is not None else
                     self.hand_pose)

        apply_trans = transl is not None or hasattr(self, 'transl')
        if transl is None:
            if hasattr(self, 'transl'):
                transl = self.transl

        if self.use_pca:
            hand_pose = torch.einsum(
                'bi,ij->bj', [hand_pose, self.hand_components])

        full_pose = torch.cat([global_orient, hand_pose], dim=1)
        full_pose += self.pose_mean

        vertices, joints = lbs(betas, full_pose, self.v_template,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, pose2rot=True,
                               )

        # Add pre-selected extra joints that might be needed
        joints = self.vertex_joint_selector(vertices, joints) # this line is commented in smplx package

        # Reorder to OpenPose-style 21-joint convention (matches HaMeR).
        joints = joints[:, self.joint_map, :]

        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if apply_trans:
            joints = joints + transl.unsqueeze(dim=1)
            vertices = vertices + transl.unsqueeze(dim=1)

        output = MANOOutput(vertices=vertices if return_verts else None,
                            joints=joints if return_verts else None,
                            betas=betas,
                            global_orient=global_orient,
                            hand_pose=hand_pose,
                            full_pose=full_pose if return_full_pose else None)

        return output
