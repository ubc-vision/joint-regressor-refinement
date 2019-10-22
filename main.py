import numpy as np
from numpy import linalg as LA
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import glob
import os
import args
from pyquaternion import Quaternion
import random
import pickle
# from pykalman import KalmanFilter


def process_video(file_name):

	right_knee = 0
	left_knee = 1
	pelvis = 2
	left_hip = 3
	right_hip = 4
	left_foot = 5
	right_foot = 6
	hair = 7


	limbs = (((left_foot, left_knee), 
			(left_knee, left_hip), 
			(left_hip, pelvis), 
			(right_foot, right_knee), 
			(right_knee, right_hip), 
			(right_hip, pelvis)))





	# with open("video_output/{}".format(file_name), 'rb') as file:
	# 	initial_point_cloud = pickle.load(file)[:-10]
	initial_point_cloud = np.load("video_output/{}".format(file_name))[2:]



	def find_joint_point(array, frame, joint):

		this_sum = np.zeros(3)
		count = 0

		for x in np.nonzero(joints[joint])[0]:
			this_sum += array[frame, x]
			count += 1

		return this_sum/count




	def savitzky_golay(y, window_size, order, deriv=0, rate=1):

		try:
			window_size = np.abs(np.int(window_size))
			order = np.abs(np.int(order))
		except (ValueError, msg):
			raise ValueError("window_size and order have to be of type int")
		if window_size % 2 != 1 or window_size < 1:
			raise TypeError("window_size size must be a positive odd number")
		if window_size < order + 2:
			raise TypeError("window_size is too small for the polynomials order")
		order_range = range(order+1)
		half_window = (window_size -1) // 2
		# precompute coefficients
		b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
		m = np.linalg.pinv(b).A[deriv] * rate**deriv * math.factorial(deriv)
		# pad the signal at the extremes with
		# values taken from the signal itself
		firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
		lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
		y = np.concatenate((firstvals, y, lastvals))
		return np.convolve( m[::-1], y, mode='valid')



	def find_forward_vector_per_frame(array, frame):

		hip_vector = (array[frame, right_hip] - array[frame, left_hip])/LA.norm(array[frame, right_hip] - array[frame, left_hip])
		up_vector = (array[frame, hair] - array[frame, pelvis])/LA.norm(array[frame, hair] - array[frame, pelvis])
		forward_vector = np.cross(up_vector, hip_vector)
		return forward_vector


	def make_person_face_forward(array):

		forward_array = np.zeros(array.shape)
		
		first_frame_forward = find_forward_vector_per_frame(array, 0)

		for frame in range(1, array.shape[0]):
			current_frame_forward = find_forward_vector_per_frame(array, frame)
			q = find_quaternion(current_frame_forward, first_frame_forward)
			for joint in [pelvis, right_hip, right_knee, left_foot, right_foot, left_hip, left_knee, hair]:
				forward_array[frame, joint] = q.rotate(array[frame, joint])

		return forward_array

	def move_to_centre(array, point_cloud):
		moved_array = np.zeros(array.shape)
		moved_point_cloud = np.zeros(point_cloud.shape)

		for frame in range(point_cloud.shape[0]):

			for point in range(array.shape[1]):
				moved_array[frame, point] = array[frame, point]-array[frame, pelvis]
			for point in range(point_cloud.shape[1]):
				moved_point_cloud[frame, point] = point_cloud[frame, point]-array[frame, pelvis]

		return moved_array, moved_point_cloud

	

	#maybe before you start remove frames that are off
	def persons_height(array):
		
		left_leg_length = (LA.norm(array[:, left_foot]-array[:, left_knee], axis = 1) + 
				LA.norm(array[:, left_knee]-array[:, left_hip], axis = 1))
		right_leg_length = (LA.norm(array[:, right_foot]-array[:, right_knee], axis = 1) + 
				LA.norm(array[:, right_knee]-array[:, right_hip], axis = 1))

		#average height of leg limbs
		#maybe get the average across the entire set of data
		height = (left_leg_length + right_leg_length)/2

		left_leg_length += LA.norm(array[:, left_hip] - array[:, pelvis], axis = 1)
		right_leg_length += LA.norm(array[:, right_hip] - array[:, pelvis], axis = 1)

		height += LA.norm(array[:, pelvis]-array[:, hair], axis = 1)
		# height += LA.norm(array[:, stomach]-array[:, chest], axis = 1)
		# height += LA.norm(array[:, chest]-array[:, hair], axis = 1)

		print(height.shape)
		print(np.std(height))
		print(np.mean(height))

		average_height = np.mean(height)
		average_left_leg_length = np.mean(left_leg_length)
		average_right_leg_length = np.mean(right_leg_length)

		return average_height, average_left_leg_length, average_right_leg_length



		


	def find_quaternion(v1, v2):

		values = np.zeros(4)
		a = np.cross(v1, v2)
		values[1:] = a
		values[0] = math.sqrt((LA.norm(v1)**2) * (LA.norm(v2)**2))+np.dot(v1, v2)
		values = values/LA.norm(values)
		q = Quaternion(values)
		return q


	#rotates vector a theta degrees around b
	def rotate_vector(a, b, theta):
		return a*math.cos(theta) + (np.cross(b, a))*math.sin(theta) + b*np.dot(b, a)*(1-math.cos(theta))

	#a onto b
	def vector_projection(a, b):
		return b*np.dot(a, b)/np.dot(b, b)




	def find_which_leg_moving_forward(array):

		array = make_person_face_forward(array)


		last_frame_left_foot_pos = array[0, left_foot]
		last_frame_right_foot_pos = array[0, right_foot]

		left_foot_forward = np.zeros(array.shape[0])

		# for frame in range(array.shape[0]):
		for frame in range(array.shape[0]):
			forward_vector = find_forward_vector_per_frame(array, frame)

			left_foot_pos = array[frame, left_foot]
			right_foot_pos = array[frame, right_foot]

			current_difference = left_foot_pos - right_foot_pos

			previous_difference = last_frame_left_foot_pos - last_frame_right_foot_pos

			current_difference = vector_projection(current_difference, forward_vector)
			previous_difference = vector_projection(previous_difference, forward_vector)

			#project both of these onto the forward vector


			
			left_foot_forward[frame] = np.dot(current_difference-previous_difference, forward_vector)

			last_frame_left_foot_pos = left_foot_pos
			last_frame_right_foot_pos = right_foot_pos

		fig = plt.figure()
		plt.plot(left_foot_forward)

		left_foot_forward = savitzky_golay(left_foot_forward, 251, 3)


		print("saving _relative_foot_movement")
		
		plt.plot(left_foot_forward)
		plt.ylabel('left foot forward')
		plt.savefig('video_output/{}_relative_foot_movement.png'.format(file_name[:-4]))
		# plt.show()
		plt.close()

		return left_foot_forward

	#TODO might have a problem if the directions change even number of times succesively
	# meaning that the foot actually did change directions
	# maybe remove all of these if this is the case
	# maybe instead of taking the average, remove directions entirely that only last a couple frames
	# or only average frames that come in odd numbers

	#make sure strides take all this into account
	def draw_animation(array, point_cloud):

		bool_left_foot_forward = find_which_leg_moving_forward(array)
		strides = find_stride_length(array)

		fig = plt.figure()
		ax = Axes3D(fig)


		
		max_val = np.amax(array)
		min_val = np.amin(array)

		prev_stride = strides[0]
		print("found stride lengths")
		for x in range(1, len(strides)):
			if(strides[x]!= prev_stride):
				print(strides[x])

			prev_stride = strides[x]



		def draw_frame(frame):
			ax.clear()
			ax.set_xlim3d(-.4, .4)
			ax.set_ylim3d(-.8, .8)
			ax.set_zlim3d(-.6, 0.6)
			ax.set_xlabel('X Label')
			ax.set_ylabel('Y Label')
			ax.set_zlabel('Z Label')
			ax.view_init(elev=90, azim=0)
			# ax.text2D(0.05, 0.95, "stride length {} cm\nframe number {}".format(strides[frame], frame), transform=ax.transAxes)
			ax.text2D(0.05, 0.95, "stride length {} cm".format(strides[frame]), transform=ax.transAxes)
			print("processing frame", frame)


			#only draws 1/10th of the points
			#also draws the same points from frame to frame
			for key_point in range(0, point_cloud.shape[1], 10):
				ax.scatter(point_cloud[frame, key_point, 0], point_cloud[frame, key_point, 1], point_cloud[frame, key_point, 2], c='g')

			for joint_a, joint_b in limbs:
				color = 'b'
				#right foot
				if(joint_a == left_foot):
					if(bool_left_foot_forward[frame] > 0):
						color = 'r'
				#left foot
				elif(joint_a == right_foot):
					if(bool_left_foot_forward[frame] < 0):
						color = 'r'
				ax.plot([array[frame, joint_a, 0], array[frame, joint_b, 0]], [array[frame, joint_a, 1], array[frame, joint_b, 1]], [array[frame, joint_a, 2], array[frame, joint_b, 2]],color = color)


		# draw_frame(1770)

		# anim = FuncAnimation(fig, draw_frame, frames= array.shape[0], interval=30)
		anim = FuncAnimation(fig, draw_frame, frames=range(0, array.shape[0], 10), interval=100)
		anim.save('video_output/{}.mp4'.format(file_name[:-4]), dpi=80)
		# plt.show()
		plt.close()
		

	def plot_limbs(array, normalize = False):

		fig = plt.figure()

		if normalize:

			limb_length = find_size_normalization(array)

			plt.plot(limb_length)

			limb_length = find_size_normalization(array, smoothed = True)

			plt.plot(limb_length)

			plt.ylabel('normalized length')

			plt.savefig('video_output/{}_normalized.png'.format(file_name[:-4]))
			plt.close()

		else:

			limb_length = return_limb_lengths(array, normalize = normalize)

			iterator = 0
			for limb in limbs:
				plt.plot(limb_length[:, iterator])
				iterator += 1

			plt.ylabel('limb lengths')
			plt.savefig('video_output/{}.png'.format(file_name[:-4]))
			plt.close()

	def return_limb_lengths(array, normalize = False):

		limb_length = np.zeros((array.shape[0], len(limbs)))

		iterator = 0
		for joint_a, joint_b in limbs:

			for frame in range(array.shape[0]):
				limb_length[frame, iterator] = LA.norm(array[frame, joint_a]-array[frame, joint_b])
				#  math.sqrt((array[frame, joint_a, 0]-array[frame, joint_b, 0])**2 + (array[frame, joint_a, 1]-array[frame, joint_b, 1])**2 + (array[frame, joint_a, 2]-array[frame, joint_b, 2])**2)
			
			# plt.plot(limb_length[:, iterator]/np.average(limb_length[:, iterator]))

			limb_length[:, iterator] = limb_length[:, iterator]/np.average(limb_length[:, iterator])

			iterator += 1

		if normalize:
			for frame in range(array.shape[0]):
				limb_length[frame, :] = limb_length[frame, :]/np.average(limb_length[frame, :])
		return limb_length


	def leg_lengths(array):
		fig = plt.figure()
		

		left_leg_lengths = np.zeros(array.shape[0])
		right_leg_lengths = np.zeros(array.shape[0])

		for frame in range(array.shape[0]):
			left_leg_lengths[frame] = (LA.norm(array[frame, left_foot]-array[frame, left_knee]) + 
					LA.norm(array[frame, left_knee]-array[frame, left_hip]) + 
					LA.norm(array[frame, left_hip]-array[frame, pelvis]))
			right_leg_lengths[frame] = (LA.norm(array[frame, right_foot]-array[frame, right_knee]) + 
					LA.norm(array[frame, right_knee]-array[frame, right_hip]) + 
					LA.norm(array[frame, right_hip]-array[frame, pelvis]))


		plt.plot(left_leg_lengths)
		plt.plot(right_leg_lengths)

		# left_leg_lengths = savitzky_golay(left_leg_lengths, 151, 3)
		# right_leg_lengths = savitzky_golay(right_leg_lengths, 151, 3)

		# plt.plot(left_leg_lengths)
		# plt.plot(right_leg_lengths)
		plt.savefig('video_output/{}_leg_lengths.png'.format(file_name[:-4]))
		plt.close()
		return left_leg_lengths, right_leg_lengths

	def find_size_normalization(array, smoothed = False):
		limb_lengths = return_limb_lengths(array)

		lengths_by_frame = np.mean(limb_lengths, axis = 1)

		if(smoothed):

			lengths_by_frame = savitzky_golay(lengths_by_frame, 151, 3)


		return lengths_by_frame

	# #only normalize size based on the legs. 
	# def normalize_size(array):

	# 	normalized_array = np.zeros(array.shape)

	# 	sizes = find_size_normalization(array, smoothed = True)
	# 	for frame in range(array.shape[0]):
	# 		normalized_array[frame] = array[frame] / sizes[frame]

	# 	return normalized_array


	def find_frames_changing_direction(array):

		frames = []

		bool_left_foot_forward = find_which_leg_moving_forward(array)

		#every time there is a switch in foot forward position, 
		#  get the distance to the previous foot position

		last_frame_changed = 0
		for frame in range(1, len(bool_left_foot_forward)):

			#if they are not the same
			if (bool_left_foot_forward[frame] * bool_left_foot_forward[frame-1] < 0):

				frames.append(frame)


		print("frames changed before")
		print(frames)

		def remove_closest_switches():

			smallest_number = args.args.direction_change_threshold
			index = -1

			for frame in range(1, len(frames)):
				this_diff = abs(frames[frame]-frames[frame-1])

				if(this_diff<smallest_number):
					index = frame
					smallest_number = this_diff

			return index

		while True:
			index = remove_closest_switches()

			if(index == -1):
				break

			del frames[index]
			del frames[index-1]

		#wow, so hacky
		for _ in range(5):
			strides = np.zeros(len(frames))

			height, left_leg_length, right_leg_length = persons_height(array)
			average_leg_length = (left_leg_length + right_leg_length)/2
			left_leg_lengths, right_leg_lengths = leg_lengths(array)
			actual_height = args.args.height
			conversion = actual_height / height

			for frame in range(1, len(frames)):

				this_frame = frames[frame]
				prev_frame = frames[frame-1]



				this_left_foot_pos = array[this_frame, pelvis] + (array[this_frame, left_foot]-array[this_frame, pelvis])/left_leg_lengths[this_frame]*average_leg_length
				this_right_foot_pos = array[this_frame, pelvis] + (array[this_frame, right_foot]-array[this_frame, pelvis])/right_leg_lengths[this_frame]*average_leg_length
				last_frame_left_foot_pos = array[prev_frame, pelvis] + (array[prev_frame, left_foot]-array[prev_frame, pelvis])/left_leg_lengths[prev_frame]*average_leg_length
				last_frame_right_foot_pos = array[prev_frame, pelvis] + (array[prev_frame, right_foot]-array[prev_frame, pelvis])/right_leg_lengths[prev_frame]*average_leg_length
				strides[frame] = conversion*(LA.norm((this_left_foot_pos-last_frame_left_foot_pos) - (this_right_foot_pos-last_frame_right_foot_pos)))

			print("strides")
			print(strides)

			#if a frame has small strides on both sides, remove it
			for this_frame in range(len(frames)-1, 1, -1):
				if(strides[this_frame] < 50 and strides[this_frame-1] < 50):
					del frames[this_frame]

		# for every frame left remove the ones that have stride less than a certain value
		# this is a bit hacky so maybe remove later


		print("frames that the leg is changing direction after smoothing")
		print(frames)

		return frames



	def find_stride_length(array):

		strides = np.zeros(array.shape[0])

		height, left_leg_length, right_leg_length = persons_height(array)

		average_leg_length = (left_leg_length + right_leg_length)/2


		left_leg_lengths, right_leg_lengths = leg_lengths(array)

		actual_height = args.args.height
		conversion = actual_height / height


		#every time there is a switch in foot forward position, 
		#  get the distance to the previous foot position

		last_frame_left_foot_pos = array[0, left_foot]
		last_frame_right_foot_pos = array[0, right_foot]

		frames_changing_direction = find_frames_changing_direction(array)

		last_frame_changed = 0

		frames_of_changing_directions = []

		for frame in frames_changing_direction:


			#get the largest distace between the 2 feet
			largest_distance = 0

			search_threshold = args.args.direction_change_threshold

			#the position it happens isnt exactly at the correct frame all the time, this check surrounding frames. 
			for this_frame in range(max(0, frame-search_threshold), min(array.shape[0], frame+search_threshold)):
				this_left_foot_pos = array[this_frame, pelvis] + (array[this_frame, left_foot]-array[this_frame, pelvis])/left_leg_lengths[this_frame]*average_leg_length
				this_right_foot_pos = array[this_frame, pelvis] + (array[this_frame, right_foot]-array[this_frame, pelvis])/right_leg_lengths[this_frame]*average_leg_length
				distance = LA.norm(this_left_foot_pos-this_right_foot_pos)
				if(distance > largest_distance):
					largest_distance = distance
					left_foot_pos = this_left_foot_pos
					right_foot_pos = this_right_foot_pos
					index = this_frame

			frames_of_changing_directions.append(index)


			#subtract the foot change vectors because they are intially pointing in different directions.
			stride = conversion*(LA.norm((left_foot_pos-last_frame_left_foot_pos) - (right_foot_pos-last_frame_right_foot_pos)))

			for prev_frame in range(frame, last_frame_changed, -1):
				strides[prev_frame] = stride


			last_frame_left_foot_pos = left_foot_pos
			last_frame_right_foot_pos = right_foot_pos
			last_frame_changed = frame


		print("frames of largest distance")
		print(frames_of_changing_directions)



		return strides

	joints = np.load("joint_locations.npy")
	base_array = np.zeros((initial_point_cloud.shape[0], 8, 3))
	for frame in range(base_array.shape[0]):
		for joint in range(base_array.shape[1]):
			base_array[frame, joint] = find_joint_point(initial_point_cloud, frame, joint)


	plot_limbs(base_array)
	plot_limbs(base_array, normalize = True)

	moved_array, moved_point_cloud = move_to_centre(base_array, initial_point_cloud)

	leg_lengths(moved_array)
	# normalized_array = normalize_size(moved_array)

	left_foot_forward = find_which_leg_moving_forward(moved_array)

	find_stride_length(moved_array)

	draw_animation(moved_array, moved_point_cloud)



process_video("slow_mo_undistorted.npy")
# for root, directory, file in os.walk("video_output/"):
# 	for file_name in file:
# 		if ".npy" in file_name:
# 			print(file_name)
# 			process_video(file_name)
