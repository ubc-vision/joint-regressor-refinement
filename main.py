import numpy as np
# from numpy import linalg as LA
# import math
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import time
import glob
# import os
# import args
# from pyquaternion import Quaternion
# import random
# import pickle

from constants import *

from array_modifier import *



def process_video(file_name):

	initial_array = np.load(file_name)

	moved_array = move_to_centre(initial_array)


	moved_array = swap_incorrect_feet(moved_array)

	# swap_incorrect_feet(moved_array)
	# find_which_leg_moving_forward(moved_array)



	# draw_animation(moved_array, file_name = file_name)

	# leg_lengths(moved_array)
	# normalized_array = normalize_size(moved_array)

	strides, frames_of_changing_directions = find_stride_length(moved_array, file_name = file_name)

	height, left_leg_length, right_leg_length = persons_height(moved_array)
	initial_average_leg_length = (left_leg_length + right_leg_length)/2
	actual_height = args.args.height
	initial_conversion = actual_height / height
	#find the average length of the leg at this point
	# find_which_leg_moving_forward(moved_array, frames_of_changing_directions)


	print()
	print("stride length of initial array")
	print()
	print("frames of largest distance")
	print(frames_of_changing_directions)
	prev_stride = strides[0]
	print("found stride lengths")
	for x in range(1, len(strides)):
		if(strides[x]!= prev_stride):
			print("%.3f" % strides[x], end="\t")
		prev_stride = strides[x]
	print()


	# find_stride_length(moved_array)

	

	smoothed_array = smooth_values(moved_array, file_name = file_name)


	strides, frames_of_changing_directions = find_stride_length(smoothed_array, 
																smooth = False, 
																average_leg_length=initial_average_leg_length, 
																conversion = initial_conversion)

	find_which_leg_moving_forward(smoothed_array, frames_of_changing_directions, smooth = False)

	plot_limbs(smoothed_array, file_name = file_name)
	plot_limbs(smoothed_array, file_name = file_name, normalize = True)

	draw_animation(smoothed_array, smooth = False, elev = 0, azim = 0, file_name = file_name)
	draw_animation(smoothed_array, smooth = False, elev = 90, azim = 0, file_name = file_name)
	draw_animation(smoothed_array, smooth = False, elev = 180, azim = 90, file_name = file_name)
	# draw_animation(smoothed_array, smooth = False, elev = 90, azim = 0, file_name = file_name)
	print()
	print("stride length of smoothed array")
	print()
	print("frames of largest distance")
	print(frames_of_changing_directions)

	

	prev_stride = strides[0]
	print("found stride lengths")
	for x in range(1, len(strides)):
		if(strides[x]!= prev_stride):
			print("%.3f" % strides[x], end="\t")

		prev_stride = strides[x]
	print()



# process_video("different_angles/VID_20191110_084755_undistorted.npy")
for file in sorted(glob.glob("different_angles/*_undistorted.npy")):
	print(file)
	process_video(file)