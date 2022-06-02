from turtle import color
import numpy as np
from numpy import genfromtxt, int64
import matplotlib.pyplot as plt
import open3d as o3d
import os
# pcd is a xyz point cloud in n*3 shape
def genGridmap(pcd,grid_size):
    grid=np.zeros((grid_size,grid_size))
    scalex=float(grid_size-1)/float(max(pcd[:,0])-min(pcd[:,0]))
    offsetx=min(pcd[:,0])
    scaley=float(grid_size-1)/float(max(pcd[:,1])-min(pcd[:,1]))
    offsety=min(pcd[:,1])
    for i in range(pcd.shape[0]):
        x,y=pcd[i,0],pcd[i,1]
        x_scaled=int(scalex*(x-offsetx))
        y_scaled=int(scaley*(y-offsety))
        grid[x_scaled,y_scaled]+=1
    grid=grid*(255/np.max(grid))
    return grid,scalex,offsetx,scaley,offsety

def trajtogrid(traj,scalex,offsetx,scaley,offsety):
    traj_=np.zeros((traj.shape[0],2),dtype=int64)
    for i in range(traj.shape[0]):
        x,y=traj[i,0],traj[i,1]
        x_scaled=int(scalex*(x-offsetx))
        y_scaled=int(scaley*(y-offsety))
        traj_[i,0],traj_[i,1]=x_scaled,y_scaled
    return traj_

grid_size=480
pcd=genfromtxt('/home/fengkai/ROB590/minicheetah-traversability-irl/scripts/data/tare_campus/Overall_map/4310525000000.csv', delimiter=',')
index=0

grid,scalex,offsetx,scaley,offsety=genGridmap(pcd,grid_size)
traj=genfromtxt('/home/fengkai/ROB590/minicheetah-traversability-irl/scripts/data/tare_campus_2/Trajectory/5543036000000.csv', delimiter=',')
traj=trajtogrid(traj,scalex,offsetx,scaley,offsety)
temp=[]

for i in range(traj.shape[0]-1):
    if (traj[i][0]==traj[i+1][0] and  traj[i][1]==traj[i+1][1]):
        continue
    elif abs(traj[i][0]-traj[i+1][0])==1 and abs(traj[i][1]-traj[i+1][1])==1:
        temp.append(traj[i])
        temp.append(np.array((traj[i][0],traj[i+1][1])))
    elif abs(traj[i][0]-traj[i+1][0])>1:
        temp.append(traj[i])
        temp.append(np.array((int(traj[i][0]+traj[i+1][0])/2,traj[i+1][1])))
    elif abs(traj[i][1]-traj[i+1][1])>1:
        temp.append(traj[i])
        temp.append(np.array((traj[i][0],int(traj[i][1]+traj[i+1][1])/2)))

    else:
        temp.append(traj[i])
traj=np.array(temp)
temp=[]

for i in range(traj.shape[0]-1):
    if (traj[i][0]==traj[i+1][0] and  traj[i][1]==traj[i+1][1]):
        continue
    elif abs(traj[i][0]-traj[i+1][0])==1 and abs(traj[i][1]-traj[i+1][1])==1:
        temp.append(traj[i])
        temp.append(np.array((traj[i][0],traj[i+1][1])))
    elif abs(traj[i][0]-traj[i+1][0])>1:
        temp.append(traj[i])
        temp.append(np.array((int(traj[i][0]+traj[i+1][0])/2,int(traj[i+1][1]))))
    elif abs(traj[i][1]-traj[i+1][1])>1:
        temp.append(traj[i])
        temp.append(np.array((int(traj[i][0]),int(traj[i][1]+traj[i+1][1])/2)))

    else:
        temp.append(traj[i])
traj=np.array(temp)
traj=traj.astype(int)
# TEST TRAJ FORMAT
# for i in range(traj.shape[0]-1): 
#     if abs(traj[i][0]-traj[i+1][0])==1 and abs(traj[i][1]-traj[i+1][1])==1:
#         print(traj[i],traj[i+1])
#     elif abs(traj[i][0]-traj[i+1][0])>1 or abs(traj[i][1]-traj[i+1][1])>1:
#         print(traj[i],traj[i+1])
#     elif (traj[i][0]==traj[i+1][0] and  traj[i][1]==traj[i+1][1]):
#         print(traj[i],traj[i+1])
print(traj.shape)

# data_len=81
# for i in range(data_len):
#     size=80
#     center=traj[i*100+50]
#     local_map=np.zeros((size,size))
#     m=grid[center[0]-40:center[0]+40,center[1]-40:center[1]+40]
#     x,y=m.shape
#     local_map[:x,:y]=m
#     past_traj=traj[i*100:i*100+50]
#     past_traj[:,0]-=center[0]-40
#     past_traj[:,1]-=center[1]-40
#     past_traj=past_traj[past_traj[:,0]>0]
#     past_traj=past_traj[past_traj[:,0]<80]
#     past_traj=past_traj[past_traj[:,1]>0]
#     past_traj=past_traj[past_traj[:,1]<80]
#     future_traj=traj[i*100+50:i*100+100]
#     future_traj[:,0]-=center[0]-40
#     future_traj[:,1]-=center[1]-40
#     future_traj=future_traj[future_traj[:,0]>0]
#     future_traj=future_traj[future_traj[:,0]<80]
#     future_traj=future_traj[future_traj[:,1]>0]
#     future_traj=future_traj[future_traj[:,1]<80]
#     plt.figure()
#     plt.imshow(local_map)
#     plt.scatter(past_traj[:,1],past_traj[:,0],color="red")
#     plt.scatter(future_traj[:,1],future_traj[:,0],color="green")
#     # plt.show()

#     os.makedirs("/home/fengkai/ROB590/minicheetah-traversability-irl/scripts/train_data/"+str(i+54), exist_ok=True)
#     plt.savefig("/home/fengkai/ROB590/minicheetah-traversability-irl/scripts/train_data/"+str(i+54)+"/map.png")
#     np.savetxt("/home/fengkai/ROB590/minicheetah-traversability-irl/scripts/train_data/"+str(i+54)+"/map.csv",local_map)
#     np.savetxt("/home/fengkai/ROB590/minicheetah-traversability-irl/scripts/train_data/"+str(i+54)+"/past_traj.csv",past_traj)
#     np.savetxt("/home/fengkai/ROB590/minicheetah-traversability-irl/scripts/train_data/"+str(i+54)+"/future_traj.csv",future_traj)
