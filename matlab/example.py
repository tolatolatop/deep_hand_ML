import os
import numpy as np
import scipy.io as sio 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.axes3d import Axes3D
import convert_tools as convert

dataset_dir = './train/'
image_index = 1;
kinect_index = 1;
py_image_index = image_index - 1;
py_kinect_index = kinect_index - 1;
filename_prefix = "%d_%07d.png" % (kinect_index,image_index)

rgb = mpimg.imread( dataset_dir +'rgb_'+ filename_prefix)
plt.figure()
plt.imshow(rgb)
#plt.show()

synthdepth = mpimg.imread(dataset_dir+'synthdepth_' + filename_prefix)
synthdepth = synthdepth * 255
synthdepth = synthdepth.astype(np.uint16)
synthdepth = synthdepth[:,:,2] + (synthdepth[:,:,1] << 8)
plt.figure()
ind = np.where(synthdepth > 0)
plt.imshow(synthdepth, vmin=np.min(synthdepth[ind]-10), vmax=np.max(synthdepth[ind]+10))

depth = mpimg.imread(dataset_dir+'depth_' + filename_prefix)
depth = depth * 255
depth = depth.astype(np.uint16)
depth = depth[:,:,2] + (depth[:,:,1] << 8)
fig = plt.figure()
plt.imshow(depth)


joints = sio.loadmat(dataset_dir + "joint_data.mat")
joint_xyz = joints['joint_xyz']
joint_uvd = joints['joint_uvd']
jnt_xyz = joint_xyz[kinect_index-1,image_index-1,:,:]
jnt_uvd = joint_uvd[kinect_index-1,image_index-1,:,:]
jnt_colors = np.random.rand(jnt_uvd.shape[0],3)
area = np.ones(jnt_uvd.shape[0])*20
fig.hold(True)
plt.scatter(jnt_uvd[:,0],jnt_uvd[:,1],s=area,c=jnt_colors)

uvd = convert.convert_depth_to_uvd(depth)
xyz = convert.convert_uvd_to_xyz(uvd)

decimation = 4
xyz_decimated = xyz[:,0::decimation,0::decimation]

points = np.reshape(xyz_decimated,[3,xyz_decimated.shape[1]*xyz_decimated.shape[2]])

body_points = points[:,np.where(points[2,:] < 2000)]
body_points = body_points.reshape(3,body_points.shape[2])
axis_bounds = [min(body_points[0,:]), max(body_points[0,:]), min(body_points[2,:]),max(body_points[2,:]),min(body_points[1,:]),max(body_points[1,:])]
fig = plt.figure()
ax = Axes3D(fig)

ax.plot(body_points[0,:], body_points[2,:], body_points[1,:],marker='.',markersize = 1.5, linestyle = "")
ax.axis('equal')
#ax.scatter(x, y, z)
# how to set view and axis

for i in range(0,2):
	fig = plt.figure()
	ax = Axes3D(fig,[0.35,0.1,0.3,0.8])
	if i == 0:
		uvd = convert.convert_depth_to_uvd(depth)
	else:
		uvd = convert.convert_depth_to_uvd(synthdepth)
	xyz = convert.convert_uvd_to_xyz(uvd)
	points = np.reshape(xyz, [3,xyz.shape[1]*xyz.shape[2]])
	# colors
	jnt_uvd_t = np.zeros([jnt_uvd.shape[1],jnt_uvd.shape[0]])
	jnt_uvd_t[0,:] = jnt_uvd[:,0]
	jnt_uvd_t[1,:] = jnt_uvd[:,1]
	jnt_uvd_t[2,:] = jnt_uvd[:,2]
	hand_points = convert.convert_uvd_to_xyz(jnt_uvd_t.reshape(3,1,jnt_uvd_t.shape[1]))
	hand_points = hand_points.reshape(3,hand_points.shape[2])
	axis_bounds = [
	np.min(hand_points[0,:]) -20,
	np.max(hand_points[0,:]) +20,
	np.min(hand_points[2,:]) -20,
	np.max(hand_points[2,:]) +20,
	np.min(hand_points[1,:]) -20,
	np.max(hand_points[1,:]) +20]
	ipnts = np.where((points[0,:] >= axis_bounds[0])&(points[2,:] >= axis_bounds[2])&(points[1,:] >= axis_bounds[4])&(points[0,:] <= axis_bounds[1])&(points[2,:] <= axis_bounds[3])&(points[1,:] <= axis_bounds[5]))
	points = points[:,ipnts]
	points = points.reshape(3,points.shape[-1])
	ax.plot(points[0,:],points[2,:],points[1,:],linestyle="",marker=".",markersize=1.5)
	ax.scatter(hand_points[0,:],hand_points[2,:],hand_points[1,:],s=area,c=jnt_colors)
	
	convert.setAxis(axis_bounds,ax)
	ax.axis('equal')

plt.show()
