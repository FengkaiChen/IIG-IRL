# import pptk

# xyz = pptk.rand(100, 3)
# v = pptk.viewer(xyz)
# v.set(point_size=0.005)
import numpy as np
import open3d as o3d
from numpy import genfromtxt
import matplotlib.pyplot as plt
xyz = genfromtxt('/home/fengkai/ROB590/minicheetah-traversability-irl/scripts/data/tare_garage/Registered_scan/97450000000.csv', delimiter=',')



x,y,z=[],[],[]
for i in range(xyz[:,2].shape[0]):
    if xyz[i,2]>1e-1:
        x.append(xyz[i,0])
        y.append(xyz[i,1])
        z.append(xyz[i,2])
new_xyz=[]
new_xyz.append(x)
new_xyz.append(y)
new_xyz.append(z)
new_xyz=np.array(new_xyz)
new_xyz=new_xyz.T
print(xyz.shape,new_xyz.shape)
# print(xyz[:,0],xyz[:,1],xyz[:,2])
pcd = o3d.geometry.PointCloud()

 
pcd.points = o3d.utility.Vector3dVector(xyz)
# print(pcd)
# # o3d.io.write_point_cloud("./data.ply", pcd)
o3d.visualization.draw_geometries([pcd])
# plt.scatter(xyz[:,0],xyz[:,1])
# plt.show()