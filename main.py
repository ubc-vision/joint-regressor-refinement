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
# from pykalman import KalmanFilter


def process_video(file_name):

	pelvis = 0
	right_hip = 1
	right_knee = 2
	right_foot = 3
	left_hip = 4
	left_knee = 5
	left_foot = 6
	stomach = 7
	chest = 8
	face = 9
	hair = 10
	left_shoulder = 11
	left_elbow = 12
	left_hand = 13
	right_shoulder = 14
	right_elbow = 15
	right_hand = 16


	initial_array = np.load("video_output/{}".format(file_name))
	limbs = ((left_foot, left_knee), 
			(left_knee, left_hip), 
			(left_hip, pelvis), 
			(right_foot, right_knee), 
			(right_knee, right_hip), 
			(right_hip, pelvis), 
			(pelvis, stomach), 
			(stomach, chest), 
			(chest, face), 
			(face, hair), 
			(left_hand, left_elbow), 
			(left_elbow, left_shoulder), 
			(left_shoulder, chest), 
			(right_hand, right_elbow), 
			(right_elbow, right_shoulder), 
			(right_shoulder, chest))


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
		up_vector = (array[frame, stomach] - array[frame, pelvis])/LA.norm(array[frame, stomach] - array[frame, pelvis])
		forward_vector = np.cross(up_vector, hip_vector)
		return forward_vector


	def make_person_face_forward(array):

		forward_array = np.zeros(array.shape)

		

		first_frame_forward = find_forward_vector_per_frame(array, 0)


		for frame in range(1, array.shape[0]):
			current_frame_forward = find_forward_vector_per_frame(array, frame)
			q = find_quaternion(current_frame_forward, first_frame_forward)
			for joint in range(17):
				forward_array[frame, joint] = q.rotate(array[frame, joint])

		return forward_array

	

	#maybe before you start remove frames that are off
	def persons_height(array):


		# IMPORTANT
		# these conversions are just for me. It might differ person to person
		#this is because the points only represent the ankle and the middle of the face
		height_multiplier = 1.13649012535/2
		#this is going to be different. Very rough estimate for now

		# TODO 
		# change the actual positions of the head and feet to match the differences between the angle and the bottom of the feet
		# and the face point and the top of the head. 
		# maybe you only need to get the length of the leg. 

		height = 0

		#average height of leg limbs
		#maybe get the average across the entire set of data
		height += (LA.norm(array[:, left_foot]-array[:, left_knee], axis = 1)+ LA.norm(array[:, right_foot]-array[:, right_knee], axis = 1))/2
		height += (LA.norm(array[:, left_knee]-array[:, left_hip], axis = 1)+ LA.norm(array[:, right_knee]-array[:, right_hip], axis = 1))/2
		height += LA.norm(array[:, pelvis]-array[:, stomach], axis = 1)
		height += LA.norm(array[:, stomach]-array[:, chest], axis = 1)
		height += LA.norm(array[:, chest]-array[:, hair], axis = 1)

		print(height.shape)
		print(np.std(height))
		print(np.mean(height))

		average_height = np.mean(height)

		return average_height/height_multiplier

	def move_ankles_to_feet(array):
		moved_ankle_array = np.copy(array)

		leg_multiplier = 1.27036126233



		for frame in range(moved_ankle_array.shape[0]):
			right_shin = moved_ankle_array[frame, right_foot] - moved_ankle_array[frame, right_knee]
			left_shin = moved_ankle_array[frame, left_foot] - moved_ankle_array[frame, left_knee]

			moved_ankle_array[frame, right_foot] = moved_ankle_array[frame, right_knee] + right_shin*leg_multiplier
			moved_ankle_array[frame, left_foot] = moved_ankle_array[frame, left_knee] + left_shin*leg_multiplier

		return moved_ankle_array



		


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


		left_foot_forward = savitzky_golay(left_foot_forward, 11, 2)

		fig = plt.figure()
		plt.plot(left_foot_forward)
		plt.ylabel('left foot forward')
		plt.savefig('video_output/{}_relative_foot_movement.png'.format(file_name[:-4]))

		return left_foot_forward

	def draw_animation(array):



		bool_left_foot_forward = find_which_leg_moving_forward(array)
		strides = find_stride_length(array)

		fig = plt.figure()
		ax = Axes3D(fig)


		
		max_val = np.amax(array)
		min_val = np.amin(array)


		def draw_frame(frame):
			ax.clear()
			ax.set_xlim3d(min_val, max_val)
			ax.set_ylim3d(min_val, max_val)
			ax.set_zlim3d(min_val, max_val)
			ax.set_xlabel('X Label')
			ax.set_ylabel('Y Label')
			ax.set_zlabel('Z Label')
			ax.text2D(0.05, 0.95, "stride length {} cm".format(strides[frame]), transform=ax.transAxes)
			print("processing frame", frame)


			for key_point in range(array.shape[1]):
				ax.scatter(array[frame, key_point, 0], array[frame, key_point, 1], array[frame, key_point, 2], c='g')

			for joint_a, joint_b in limbs:
				color = 'b'
				#right foot
				if(joint_a == 3):
					if(bool_left_foot_forward[frame] < 0):
						color = 'r'
				#left foot
				elif(joint_a == 6):
					if(bool_left_foot_forward[frame] > 0):
						color = 'r'
				ax.plot([array[frame, joint_a, 0], array[frame, joint_b, 0]], [array[frame, joint_a, 1], array[frame, joint_b, 1]], [array[frame, joint_a, 2], array[frame, joint_b, 2]],color = color)


		anim = FuncAnimation(fig, draw_frame, frames=array.shape[0], interval=20)
		anim.save('video_output/{}.gif'.format(file_name[:-4]), dpi=80)
		# plt.show()

	def plot_limbs(array, normalize = False):

		fig = plt.figure()

		if normalize:

			limb_length = find_size_normalization(array)

			plt.plot(limb_length)

			limb_length = find_size_normalization(array, smoothed = True)

			plt.plot(limb_length)

			plt.ylabel('normalized length')

			plt.savefig('video_output/{}_normalized.png'.format(file_name[:-4]))

		else:

			limb_length = return_limb_lengths(array, normalize = normalize)

			iterator = 0
			for limb in limbs:
				plt.plot(limb_length[:, iterator])
				iterator += 1

			plt.ylabel('limb lengths')
			plt.savefig('video_output/{}.png'.format(file_name[:-4]))

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

	def find_size_normalization(array, smoothed = False):
		limb_lengths = return_limb_lengths(array)

		lengths_by_frame = np.mean(limb_lengths, axis = 1)

		if(smoothed):

			lengths_by_frame = savitzky_golay(lengths_by_frame, 51, 3)


		return lengths_by_frame

	def normalize_size(array):

		normalized_array = np.zeros(array.shape)

		sizes = find_size_normalization(array, smoothed = True)
		for frame in range(array.shape[0]):
			normalized_array[frame] = array[frame] / sizes[frame]

		return normalized_array





	def find_stride_length(array):

		strides = np.zeros(array.shape[0])

		height = persons_height(array)
		actual_height = args.args.height
		conversion = actual_height / height



		bool_left_foot_forward = find_which_leg_moving_forward(array)
		#every time there is a switch in foot forward position, 
		#  get the distance to the previous foot position

		last_frame_left_foot_pos = array[0, left_foot]
		last_frame_right_foot_pos = array[0, right_foot]

		last_frame_changed = 0
		for frame in range(1, len(bool_left_foot_forward)):

			#if they are not the same
			if (bool_left_foot_forward[frame] * bool_left_foot_forward[frame-1] < 0):
				left_foot_pos = array[frame-1, left_foot]
				right_foot_pos = array[frame-1, right_foot]

				#subtract the foot change vectors because they are intially pointing in the same direction.
				# it should actually get flipped over the forward vector
				stride = conversion*(LA.norm((left_foot_pos-last_frame_left_foot_pos) - (right_foot_pos-last_frame_right_foot_pos)))
	
				for prev_frame in range(frame, last_frame_changed, -1):
					strides[prev_frame] = stride


				last_frame_left_foot_pos = array[frame, left_foot]
				last_frame_right_foot_pos = array[frame, right_foot]
				last_frame_changed = frame

		return strides


	# plot_limbs(initial_array)
	# plot_limbs(initial_array, normalize = True)



	normalized_array = normalize_size(initial_array)

	moved_ankle_array = move_ankles_to_feet(normalized_array)

	left_foot_forward = find_which_leg_moving_forward(moved_ankle_array)

	draw_animation(moved_ankle_array)



# process_video("VID_20190415_200634_numpy_output.npy")
for root, directory, file in os.walk("video_output/"):
	for file_name in file:
		if ".npy" in file_name:
			print(file_name)
			process_video(file_name)
