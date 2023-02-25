import torch
num_voxel=4
num_points_in_pillar=8
num_points_in_voxel=num_points_in_pillar//num_voxel
Z=8
H=2
W=2
zs = torch.linspace(0, Z - 0, num_points_in_pillar, ).view(num_voxel,num_points_in_voxel, 1, 1).permute(1,0,2,3).expand(num_points_in_voxel,num_voxel, H, W) / Z
xs = torch.linspace(0, W - 0, W, ).view(1,1, 1, W).expand(num_points_in_voxel,num_voxel, H, W) / W
ys = torch.linspace(0, H - 0, H,).view(1, 1,H, 1).expand(num_points_in_voxel,num_voxel, H, W) / H
ref_3d = torch.stack((xs, ys, zs), -1)
print(ref_3d[0])
ref_3d = ref_3d.permute(0, 4, 1, 2, 3).flatten(2).permute(0, 2, 1)

print(ref_3d.shape)
