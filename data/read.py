import numpy as np
# import pcl 
import ros_numpy 
import rosbag
import sensor_msgs
import pptk
import numpy as np
import open3d as o3d
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import matplotlib.pyplot as plt


def convert_pc_msg_to_np(pc_msg):
    pc_msg.__class__ = sensor_msgs.msg._PointCloud2.PointCloud2
    offset_sorted = {f.offset: f for f in pc_msg.fields}
    pc_msg.fields = [f for (_, f) in sorted(offset_sorted.items())]

    # Conversion from PointCloud2 msg to np array.
    pc_np = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pc_msg, remove_nans=True)
    # pc_pcl = pcl.PointCloud(np.array(pc_np, dtype=np.float32))
    return pc_np
    # , pc_pcl  # point cloud in numpy and pcl format

# # Use a ros subscriber as you already suggested or is shown in the other
# # answers to run it online :)

# # To run it offline on a rosbag use:

for topic, msg, t in rosbag.Bag('/media/fengkai/EBAF-8BC0/example_data/bag_file/tare_campus_2.bag').read_messages():
    if topic=="/camera/image":
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        image=np.array(cv_image)
        # plt.imshow(image)
        # plt.show()
        # plt.imsave()
        plt.imsave('/home/fengkai/ROB590/minicheetah-traversability-irl/scripts/data/tare_campus_2/Image/'+str(t)+'.png', image)
        
        
        # np.savetxt('/home/fengkai/ROB590/minicheetah-traversability-irl/scripts/data/RGBimage/'+str(t)+'.csv', image, delimiter=',')
    elif topic == "/trajectory":
         pc_np = convert_pc_msg_to_np(msg)
         np.savetxt('/home/fengkai/ROB590/minicheetah-traversability-irl/scripts/data/tare_campus_2/Trajectory/'+str(t)+'.csv', pc_np, delimiter=',')

    # elif topic == "/terrain_map":
    #     pc_np = convert_pc_msg_to_np(msg)
    #     np.savetxt('/home/fengkai/ROB590/minicheetah-traversability-irl/scripts/data/room10/Terrain_map/'+str(t)+'.csv', pc_np, delimiter=',')

    elif topic == "/registered_scan":
        pc_np = convert_pc_msg_to_np(msg)
        np.savetxt('/home/fengkai/ROB590/minicheetah-traversability-irl/scripts/data/tare_campus_2/Registered_scan/'+str(t)+'.csv', pc_np, delimiter=',')
    elif topic == "/overall_map":
        pc_np = convert_pc_msg_to_np(msg)
        np.savetxt('/home/fengkai/ROB590/minicheetah-traversability-irl/scripts/data/tare_campus_2/Overall_map/'+str(t)+'.csv', pc_np, delimiter=',')
    elif topic == "/explored_volume":
        temp=np.array([float(msg.data)])
        np.savetxt('/home/fengkai/ROB590/minicheetah-traversability-irl/scripts/data/tare_campus_2/Explored_volume/'+str(t)+'.csv', temp, delimiter=',')
