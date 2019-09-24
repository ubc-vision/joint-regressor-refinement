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


	array = np.load("video_output/{}".format(file_name))
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

	

	def persons_height():
		height = 0

		#average height of leg limbs
		height += (LA.norm(array[0, left_foot]-array[0, left_knee])+ LA.norm(array[0, right_foot]-array[0, right_knee]))/2
		height += (LA.norm(array[0, left_knee]-array[0, left_hip])+ LA.norm(array[0, right_knee]-array[0, right_hip]))/2
		height += LA.norm(array[0, pelvis]-array[0, stomach])
		height += LA.norm(array[0, stomach]-array[0, chest])
		height += LA.norm(array[0, chest]-array[0, hair])

		return height/size_normalized[0]




	#rotates vector a theta degrees around b
	def rotate_vector(a, b, theta):
		return a*math.cos(theta) + (np.cross(b, a))*math.sin(theta) + b*np.dot(b, a)*(1-math.cos(theta))

	#a onto b
	def vector_projection(a, b):
		return b*np.dot(a, b)/np.dot(b, b)


	#TODO youre going to have to take into account if the person rotates while stepping
	#	this will change finding which leg is moving forward
	#	and the stride length
	def find_which_leg_moving_forward():
		

		last_frame_left_foot_pos = array[0, left_foot]
		last_frame_right_foot_pos = array[0, right_foot]

		bool_left_foot_forward = np.zeros(array.shape[0])

		# for frame in range(array.shape[0]):
		for frame in range(array.shape[0]):
			hip_vector = (array[frame, right_hip] - array[frame, left_hip])/LA.norm(array[frame, right_hip] - array[frame, left_hip])
			up_vector = (array[frame, stomach] - array[frame, pelvis])/LA.norm(array[frame, stomach] - array[frame, pelvis])
			forward_vector = np.cross(up_vector, hip_vector)

			left_foot_pos = array[frame, left_foot]
			right_foot_pos = array[frame, right_foot]

			current_difference = left_foot_pos - right_foot_pos
			previous_difference = last_frame_left_foot_pos - last_frame_right_foot_pos

			current_difference = vector_projection(current_difference, forward_vector)
			previous_difference = vector_projection(previous_difference, forward_vector)

			#project both of these onto the forward vector


			#right foot is moving forward
			if(np.dot(current_difference-previous_difference, forward_vector)<0):
				bool_left_foot_forward[frame] = 0
				# print("right foot forward")
			else:
				bool_left_foot_forward[frame] = 1
				# print("left foot forward")

			last_frame_left_foot_pos = left_foot_pos
			last_frame_right_foot_pos = right_foot_pos


		return bool_left_foot_forward

	def draw_animation():

		bool_left_foot_forward = find_which_leg_moving_forward()
		strides = find_stride_length()

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
			print(frame)


			for key_point in range(array.shape[1]):
				ax.scatter(array[frame, key_point, 0], array[frame, key_point, 1], array[frame, key_point, 2], c='g')

			for joint_a, joint_b in limbs:
				color = 'b'
				#right foot
				if(joint_a == 3):
					if(bool_left_foot_forward[frame]==0):
						color = 'r'
				#left foot
				elif(joint_a == 6):
					if(bool_left_foot_forward[frame]==1):
						color = 'r'
				ax.plot([array[frame, joint_a, 0], array[frame, joint_b, 0]], [array[frame, joint_a, 1], array[frame, joint_b, 1]], [array[frame, joint_a, 2], array[frame, joint_b, 2]],color = color)


		anim = FuncAnimation(fig, draw_frame, frames=array.shape[0], interval=33)
		anim.save('video_output/{}.mp4'.format(file_name[:-4]), dpi=80)
		# plt.show()

	def plot_limbs(normalize = False):


		limb_length = return_limb_lengths(normalize = normalize)

		iterator = 0
		for limb in limbs:
			plt.plot(limb_length[:, iterator])
			iterator += 1

		plt.ylabel('normalized length')


		if(normalize == False):
			plt.savefig('video_output/{}.png'.format(file_name[:-4]))
		else:
			plt.savefig('video_output/{}_normalized.png'.format(file_name[:-4]))

	def return_limb_lengths(normalize = False):

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

	def find_size_normalization():
		limb_lengths = return_limb_lengths()

		lengths_by_frame = np.mean(limb_lengths, axis = 1)

		return lengths_by_frame



		#find the sum along the 


	def find_stride_length():

		strides = np.zeros(array.shape[0])

		height = persons_height()
		actual_height = args.args.height
		conversion = actual_height / height



		bool_left_foot_forward = find_which_leg_moving_forward()
		#every time there is a switch in foot forward position, 
		#  get the distance to the previous foot position

		last_frame_left_foot_pos = array[0, left_foot]/size_normalized[0]
		last_frame_right_foot_pos = array[0, right_foot]/size_normalized[0]

		last_frame_changed = 0
		for frame in range(1, len(bool_left_foot_forward)):

			if (bool_left_foot_forward[frame] != bool_left_foot_forward[frame-1]):
				left_foot_pos = array[frame-1, left_foot]/size_normalized[frame-1]
				right_foot_pos = array[frame-1, right_foot]/size_normalized[frame-1]

				stride = conversion*max(LA.norm(left_foot_pos-last_frame_left_foot_pos),LA.norm(right_foot_pos-last_frame_right_foot_pos))
	
				for prev_frame in range(frame, last_frame_changed, -1):
					strides[prev_frame] = stride


				last_frame_left_foot_pos = array[frame, left_foot]/size_normalized[frame]
				last_frame_right_foot_pos = array[frame, right_foot]/size_normalized[frame]
				last_frame_changed = frame

		return strides


	# plot_limbs()
	# plot_limbs(normalize = True)
	size_normalized = find_size_normalization()
	draw_animation()



for root, directory, file in os.walk("video_output/"):
	for file_name in file:
		if ".npy" in file_name:
			print(file_name)
			process_video(file_name)