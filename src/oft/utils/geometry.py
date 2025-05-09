"""
Geometry utilities for 3D boxes: rotation und BEV-IoU.
"""

import torch
import numpy as np

def quaternion_to_rotmat(q):
    """
    q = [x, y, z, w]   (w = scalar part)
    Returns 3×3 Rotation-Matrix.
    """
    x,y,z,w = q
    xx,yy,zz = x*x, y*y, z*z
    xy,xz,yz = x*y, x*z, y*z
    wx,wy,wz = w*x, w*y, w*z
    return np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
    ], dtype=np.float32)


def box3d_iou_torch(box1: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """
    BEV-IoU zwischen einer Box und N Boxen.
    box1: [7] oder [1,7], boxes: [N,7]
    Jede Zeile: [x, y, z, w, l, h, yaw]
    """
    # Flatten inputs
    b1 = box1.view(-1).float()
    b2 = boxes.view(-1,7).float()
    if b1.numel() != 7:
        raise ValueError(f"box1 must have 7 elements, got {b1.numel()}")
    if b2.dim()!=2 or b2.shape[1]!=7:
        raise ValueError(f"boxes must be shape [N,7], got {tuple(b2.shape)}")

    def bev_corners(b: torch.Tensor):
        x,y,z,w,l,h,yaw = b.unbind(0)
        cy = torch.cos(yaw); sy = torch.sin(yaw)
        # vorne-links
        fl = torch.tensor([ x + w/2*cy - l/2*sy,
                            y + w/2*sy + l/2*cy ], device=b.device)
        fr = torch.tensor([ x - w/2*cy - l/2*sy,
                            y - w/2*sy + l/2*cy ], device=b.device)
        rr = torch.tensor([ x - w/2*cy + l/2*sy,
                            y - w/2*sy - l/2*cy ], device=b.device)
        rl = torch.tensor([ x + w/2*cy + l/2*sy,
                            y + w/2*sy - l/2*cy ], device=b.device)
        return torch.stack([fl, fr, rr, rl], dim=0)  # [4,2]

    c1 = bev_corners(b1)  # [4,2]
    c2_list = [bev_corners(b) for b in b2]

    min_x1,_ = torch.min(c1[:,0],dim=0); max_x1,_ = torch.max(c1[:,0],dim=0)
    min_y1,_ = torch.min(c1[:,1],dim=0); max_y1,_ = torch.max(c1[:,1],dim=0)
    area1   = (max_x1-min_x1)*(max_y1-min_y1)

    ious = []
    for c2 in c2_list:
        min_x2,_ = torch.min(c2[:,0],dim=0); max_x2,_ = torch.max(c2[:,0],dim=0)
        min_y2,_ = torch.min(c2[:,1],dim=0); max_y2,_ = torch.max(c2[:,1],dim=0)
        area2 = (max_x2-min_x2)*(max_y2-min_y2)

        inter_w = torch.clamp(torch.min(max_x1,max_x2)-torch.max(min_x1,min_x2), min=0)
        inter_h = torch.clamp(torch.min(max_y1,max_y2)-torch.max(min_y1,min_y2), min=0)
        inter  = inter_w * inter_h
        union  = area1 + area2 - inter
        ious.append(inter/union if union>0 else torch.tensor(0.0,device=b1.device))

    return torch.stack(ious)


def compute_box_corners(box) -> np.ndarray:
    """ Berechnet die Eckpunkte einer Box in Welt-Koordinaten.
    Unterstützt:
      • box.corners() → (3,8) oder (8,3)            
      • eigene Box-Objekte mit .center, .w, .l, .h, .orientation
    Rückgabe: Array (8,3)
    """
    cx,cy,cz = box.center
    w,h,l    = box.w, box.h, box.l
    x_c = np.array([ w/2,  w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2], dtype=np.float32)
    y_c = np.array([ h/2,  h/2,  h/2,  h/2, -h/2, -h/2, -h/2, -h/2], dtype=np.float32)
    z_c = np.array([ l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2,  l/2], dtype=np.float32)
    corners = np.vstack((x_c, y_c, z_c))  # (3,8)
    R = quaternion_to_rotmat(box.rotation)
    corners_rot = (R @ corners).T         # (8,3)
    return corners_rot + np.array([cx, cy, cz], dtype=np.float32)