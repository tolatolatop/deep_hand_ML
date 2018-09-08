import numpy as np 

def convert_depth_to_uvd(depth):
	x = np.arange(1,depth.shape[1] + 1)
	y = np.arange(1,depth.shape[0] + 1)
	mx,my = np.meshgrid(x,y)
	return np.vstack((mx,my,depth)).reshape([3,480,640])

def convert_uvd_to_xyz(uvd):
	xRes = 640
	yRes = 480

	xzFactor = 1.08836710
	yzFactor = 0.817612648

	normalizedX = uvd[0,:,:] / xRes - 0.5
	normalizedY = 0.5 - uvd[1,:,:] / yRes

	xyz = np.zeros(uvd.shape)
	xyz[2,:,:] = uvd[2,:,:]
	xyz[0,:,:] = normalizedX * xyz[2,:,:] * xzFactor
	xyz[1,:,:] = normalizedY * xyz[2,:,:] * yzFactor
	return xyz

def setAxis(axis_bounds,axes):
	axes.set_xlim([axis_bounds[0],axis_bounds[1]])
	axes.set_ylim([axis_bounds[2],axis_bounds[3]])
	axes.set_zlim([axis_bounds[4],axis_bounds[5]])
