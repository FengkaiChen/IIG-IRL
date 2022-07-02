import rospy
import torch
import os
# import pcl
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import ros_numpy
import numpy as np
from network.res_unet import ResUnet
def genGridmap(pcd,grid_size):
    grid=torch.zeros((grid_size,grid_size))
    scalex=float(grid_size-1)/float(max(pcd[:,0])-min(pcd[:,0]))
    offsetx=min(pcd[:,0])
    scaley=float(grid_size-1)/float(max(pcd[:,1])-min(pcd[:,1]))
    offsety=min(pcd[:,1])
    # grid_pcd=np.zeros_like(pcd)
    for i in range(pcd.shape[0]):
        x,y=pcd[i,0],pcd[i,1]
        x_scaled=int(scalex*(x-offsetx))
        y_scaled=int(scaley*(y-offsety))
        grid[x_scaled,y_scaled]+=1
        # grid[i]=x_scaled,y_scaled
    return grid,scalex,offsetx,scaley,offsety

def genlocalMap(grid_pcd,center,local_size):
    x1,y1=max(center[0]-40,0),min(center[1]-40,480)
    x2,y2=max(center[0]+40,0),min(center[1]+40,480)

    local_map=np.zeros((local_size,local_size))
    center_z=center[2]
    for i in range(grid_pcd.shape[0]):
        x,y,z=grid_pcd[i,0],grid_pcd[i,1],grid_pcd[i,2]
        if x1<=x<=x2 and y1<=y<=y2 and center_z<=z<=center_z+1:
            # print(int(x-center[0]+40),int(y-center[1]+40))
            local_map[int(x-center[0]+40-1),int(y-center[1]+40-1)]+=1
    local_map[local_map<=1]=0
    local_map[local_map>1]=1
    return local_map
 

def callback(data):
    pc = ros_numpy.numpify(data)
    points=np.zeros((pc.shape[0],3))
    points[:,0]=pc['x']
    points[:,1]=pc['y']
    points[:,2]=pc['z']
    p = (np.array(points, dtype=np.float32))
    grid_pcd,_,_,_,_=genGridmap(p,grid_size)
    grid_pcd=grid_pcd.reshape((1,1,grid_size,grid_size))
    reward=net(grid_pcd)
    print(reward.shape)

if __name__ == "__main__":
    # pre_train_weight="step90-nll0.9690091423449481-loss0.5284339189529419-total1.4974430799484253.pth"
    pre_train_weight = None
    net=ResUnet(channel=1)
    grid_size=80

    if pre_train_weight is None:
       pass
    else:
        pre_train_check = torch.load(os.path.join('exp', pre_train_weight))
        net.init_with_pre_train(pre_train_check)
    
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/registered_scan", PointCloud2, callback)
    rospy.spin()