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

import torch
import torch.optim as optim
import torch.nn as nn

from constants import *

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

	pelvis = (array[frame, LHip]+array[frame, RHip])/2

	hip_vector = (array[frame, RHip] - array[frame, LHip])/LA.norm(array[frame, RHip] - array[frame, LHip])
	up_vector = (array[frame, Head] - pelvis)/LA.norm(array[frame, Head] - pelvis)
	forward_vector = np.cross(up_vector, hip_vector)
	return forward_vector


def make_person_face_forward(array):

	forward_array = np.zeros(array.shape)
	
	first_frame_forward = np.array([ 0,  0, -1])
	

	for frame in range(0, array.shape[0]):
		current_frame_forward = find_forward_vector_per_frame(array, frame)
		q = find_quaternion(current_frame_forward, first_frame_forward)
		for joint in [RAnkle, RKnee, RHip, LHip, LKnee, LAnkle, RWrist, RElbow, RShoulder, LShoulder, LElbow, LWrist, Neck, Head, Nose, LEye, REye, REar, LEar]:
			forward_array[frame, joint] = q.rotate(array[frame, joint])

	return forward_array


def move_to_centre(array):
	moved_array = np.zeros(array.shape)

	for frame in range(array.shape[0]):

		pelvis = (array[frame, LHip]+array[frame, RHip])/2

		for point in range(array.shape[1]):
			moved_array[frame, point] = array[frame, point]-pelvis

	return moved_array



#return the persons height as an average over all frames
def persons_height(array):
	
	left_leg_length = (LA.norm(array[:, LAnkle]-array[:, LKnee], axis = 1) + 
			LA.norm(array[:, LKnee]-array[:, LHip], axis = 1))
	right_leg_length = (LA.norm(array[:, RAnkle]-array[:, RKnee], axis = 1) + 
			LA.norm(array[:, RKnee]-array[:, RHip], axis = 1))

	height = (left_leg_length + right_leg_length)/2

	height += LA.norm((array[:, LHip]+array[:, RHip])/2-array[:, Head], axis = 1)

	average_height = np.mean(height)
	average_left_leg_length = np.mean(left_leg_length)
	average_right_leg_length = np.mean(right_leg_length)

	return average_height, average_left_leg_length, average_right_leg_length



	

#find quaternion representing rotation from v2 to v1
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




def find_which_leg_moving_forward(array, frames_change = None, smooth = True, file_name = None):

	array = make_person_face_forward(array)


	last_frame_left_foot_pos = array[0, LAnkle]
	last_frame_right_foot_pos = array[0, RAnkle]


	feet_direction = np.zeros(len(array))

	left_feet = array[:, LAnkle]
	right_feet = array[:, RAnkle]
	current_differences = left_feet-right_feet
	feet_direction = (current_differences[1:]-current_differences[:-1])[:, 2]


	if(file_name is not None):

		fig = plt.figure()
		plt.ylim(-.2, .2)
		plt.plot(feet_direction)

		plt.axhline(y=0, color='k')

		if(frames_change is not None):
			for frame in frames_change:
				plt.axvline(x=frame, color='k')

		if(smooth):
			feet_direction = savitzky_golay(feet_direction, 77, 3)
			plt.plot(feet_direction)

		plt.ylabel('left feet_direction forward')
		plt.savefig('{}_relative_foot_movement.png'.format(file_name[:-4]))

		# plt.show()
		plt.close()

	return feet_direction

#make sure strides take all this into account
def draw_animation(array, smooth = True, elev = 180, azim = 0, file_name = None):

	bool_left_foot_forward = find_which_leg_moving_forward(array, smooth=smooth, file_name = file_name)
	strides, _ = find_stride_length(array, smooth, file_name = file_name)


	
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
		ax.view_init(elev=elev, azim=azim)
		# ax.text2D(0.05, 0.95, "stride length {} cm\nframe number {}".format(strides[frame], frame), transform=ax.transAxes)
		ax.text2D(0.05, 0.95, "stride length {} cm".format(strides[frame]), transform=ax.transAxes)
		print("processing frame", frame)


		for key_point in range(0, array.shape[1]):
			ax.scatter(array[frame, key_point, 0], array[frame, key_point, 1], array[frame, key_point, 2], c='g')

		for joint_a, joint_b in limbs:
			color = 'b'
			if(frame < len(bool_left_foot_forward)):
				#right foot
				if(joint_a == LAnkle):
					if(bool_left_foot_forward[frame] > 0):
						color = 'r'
				#left foot
				elif(joint_a == RAnkle):
					if(bool_left_foot_forward[frame] < 0):
						color = 'r'
			ax.plot([array[frame, joint_a, 0], array[frame, joint_b, 0]], [array[frame, joint_a, 1], array[frame, joint_b, 1]], [array[frame, joint_a, 2], array[frame, joint_b, 2]],color = color)


	# for x in range(100):
	# 	time.sleep(1)
	# 	fig = plt.figure()
	# 	ax = Axes3D(fig)

	# 	draw_frame(70+x)

	# 	plt.show()
	# 	plt.close()




	fig = plt.figure()
	ax = Axes3D(fig)

	anim = FuncAnimation(fig, draw_frame, frames= array.shape[0], interval=40)
	# anim = FuncAnimation(fig, draw_frame, frames=range(0, array.shape[0], 10), interval=100)
	anim.save('{}_elev_{}_azim_{}.mp4'.format(file_name[:-4], elev, azim), dpi=300)
	# plt.show()
	plt.close()
	

def plot_limbs(array, file_name, normalize = False):

	fig = plt.figure()

	if normalize:

		limb_length = find_size_normalization(array)

		plt.plot(limb_length)

		limb_length = find_size_normalization(array, smoothed = True)

		plt.plot(limb_length)

		plt.ylabel('normalized length')

		plt.savefig('{}_normalized.png'.format(file_name[:-4]))
		plt.close()

	else:

		limb_length = return_limb_lengths(array, normalize = normalize)

		iterator = 0
		for limb in limbs:
			plt.plot(limb_length[:, iterator])
			iterator += 1

		plt.ylabel('limb lengths')
		plt.savefig('{}.png'.format(file_name[:-4]))
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


def leg_lengths(array, file_name=None):
	fig = plt.figure()
	

	left_leg_lengths = np.zeros(array.shape[0])
	right_leg_lengths = np.zeros(array.shape[0])

	for frame in range(array.shape[0]):
		left_leg_lengths[frame] = (LA.norm(array[frame, LAnkle]-array[frame, LKnee]) + 
				LA.norm(array[frame, LKnee]-array[frame, LHip]))
		right_leg_lengths[frame] = (LA.norm(array[frame, RAnkle]-array[frame, RKnee]) + 
				LA.norm(array[frame, RKnee]-array[frame, RHip]))


	if(file_name is not None):

		plt.plot(left_leg_lengths)
		plt.plot(right_leg_lengths)

		# left_leg_lengths = savitzky_golay(left_leg_lengths, 151, 3)
		# right_leg_lengths = savitzky_golay(right_leg_lengths, 151, 3)

		# plt.plot(left_leg_lengths)
		# plt.plot(right_leg_lengths)
		plt.savefig('{}_leg_lengths.png'.format(file_name[:-4]))
		plt.close()

	return left_leg_lengths, right_leg_lengths

def find_size_normalization(array, smoothed = False):
	limb_lengths = return_limb_lengths(array)

	lengths_by_frame = np.mean(limb_lengths, axis = 1)

	if(smoothed):

		lengths_by_frame = savitzky_golay(lengths_by_frame, 51, 3)


	return lengths_by_frame

# #only normalize size based on the legs. 
# def normalize_size(array):

# 	normalized_array = np.zeros(array.shape)

# 	sizes = find_size_normalization(array, smoothed = True)
# 	for frame in range(array.shape[0]):
# 		normalized_array[frame] = array[frame] / sizes[frame]

# 	return normalized_array


def find_frames_changing_direction(array, smooth = True, file_name = None):

	frames = []

	bool_left_foot_forward = find_which_leg_moving_forward(array, smooth = smooth, file_name = file_name)

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

	return frames



def swap_incorrect_feet(array):

	def find_relative_positions(current_differences, forward_vectors):
		dot_product = np.dot(current_differences, forward_vectors)
		change_this_frame = forward_vectors * dot_product
		relative_positions = LA.norm(change_this_frame)
		if(dot_product<0):
			relative_positions = -1*relative_positions

		return relative_positions




	left_feet = array[:, LAnkle]
	right_feet = array[:, RAnkle]
	current_differences = right_feet-left_feet

	#project the current differences onto the forward vector
	pelvi = (array[:, LHip]+array[:, RHip])/2

	hip_vector_norm = LA.norm((array[:, RHip] - array[:, LHip]), axis = 1)

	up_vector_norm = LA.norm((array[:, Head] - pelvi), axis = 1)

	hip_vectors = (array[:, RHip] - array[:, LHip])/hip_vector_norm[:, np.newaxis]

	up_vectors = (array[:, Head] - pelvi)/up_vector_norm[:, np.newaxis]

	forward_vectors = np.cross(up_vectors, hip_vectors)


	#project current_differences onto forward_vectors

	#for each index, perform its own dot product

	relative_positions = np.zeros(len(forward_vectors))
	for x in range(len(forward_vectors)):
		relative_positions[x] = find_relative_positions(current_differences[x], forward_vectors[x])


	smoothed_plot = savitzky_golay(relative_positions, 77, 3)

	differences = abs(relative_positions-smoothed_plot)

	worst_cases = np.argsort(differences)

	for x in range(len(worst_cases)-1, 0, -1):
		index = worst_cases[x]

		feet_difference = forward_vectors[index] * relative_positions[index]

		# feet_difference = left_feet[index]-right_feet[index]

		# feet_difference = vector_projection(feet_difference, forward)

		new_left_foot = left_feet[index] + feet_difference
		new_right_foot = right_feet[index] - feet_difference

		new_relative_position = find_relative_positions(new_right_foot-new_left_foot, forward_vectors[index])

		if(abs(relative_positions[index]-smoothed_plot[index]) > abs(new_relative_position-smoothed_plot[index])):

			array[index, LAnkle] = new_left_foot
			array[index, RAnkle] = new_right_foot
			relative_positions[index] = new_relative_position

			smoothed_plot = savitzky_golay(relative_positions, 77, 3)

	return array


#TODO make it so it finds the stride length for the frame where the distance to the previous leg lengths. 
def find_stride_length(array, smooth = True, average_leg_length = None, conversion = None, file_name = None):

	strides = np.zeros(array.shape[0])

	height, left_leg_length, right_leg_length = persons_height(array)
	left_leg_lengths, right_leg_lengths = leg_lengths(array, file_name = file_name)

	if(average_leg_length is None):
		average_leg_length = (left_leg_length + right_leg_length)/2
		actual_height = args.args.height
		conversion = actual_height / height


	


	#every time there is a switch in foot forward position, 
	#  get the distance to the previous foot position

	last_frame_left_foot_pos = array[0, LAnkle]
	last_frame_right_foot_pos = array[0, RAnkle]

	bool_left_foot_forward = find_which_leg_moving_forward(array, file_name = file_name)

	frames_changing_direction = find_frames_changing_direction(array, smooth = smooth, file_name = file_name)

	last_frame_changed = 0

	frames_of_changing_directions = []



	for frame in frames_changing_direction:

		#get the largest distace between the 2 feet
		largest_distance = 0
		found_index = frame

		search_threshold = args.args.direction_change_threshold


		#the position it happens isnt exactly at the correct frame all the time. this checks surrounding frames. 
		for this_frame in range(max(0, frame-search_threshold), min(array.shape[0], frame+search_threshold)):
			
			this_left_foot_pos = array[this_frame, LAnkle]/left_leg_lengths[this_frame]*average_leg_length
			this_right_foot_pos = array[this_frame, RAnkle]/right_leg_lengths[this_frame]*average_leg_length

			forward_vec = find_forward_vector_per_frame(array, this_frame)
			forward_projection = vector_projection(this_left_foot_pos-this_right_foot_pos, forward_vec)
			distance = LA.norm(forward_projection)

			if(distance > largest_distance):
				largest_distance = distance
				left_foot_pos = this_left_foot_pos
				right_foot_pos = this_right_foot_pos
				found_index = this_frame



		frames_of_changing_directions.append(found_index)


		#subtract the foot change vectors because they are intially pointing in different directions.
		stride = conversion*(LA.norm((left_foot_pos-last_frame_left_foot_pos) - (right_foot_pos-last_frame_right_foot_pos)))

		for prev_frame in range(frame, last_frame_changed, -1):
			strides[prev_frame] = stride


		last_frame_left_foot_pos = left_foot_pos
		last_frame_right_foot_pos = right_foot_pos
		last_frame_changed = frame


	return strides, frames_of_changing_directions


#make one of the losses the actual stride length
#make one of the losses the frame that the foot is changing directions. 
# is this differentiable?
def smooth_values(array, file_name = None):

	def dot_product(vec_a, vec_b):
		return torch.bmm(vec_a.view(-1, 1, 3), vec_b.view(-1, 3, 1)).view(-1, 1)


	#only use a list of values that corresponds to the known points


	_, inflection_points = find_stride_length(array, file_name = file_name)

	initial_tensor = torch.tensor(array, dtype=torch.float64, requires_grad=False)
	tensor = torch.tensor(array, dtype=torch.float64, requires_grad=True)

	
	optimizer = optim.SGD([tensor], lr=100)


	#make another loss that is the difference between the current length and the average length for each leg. 


	starting_leg_lengths = torch.ones([tensor.shape[0], len(limbs)], dtype=torch.float64, requires_grad=False)


	#differences between leg lengths
	for index, (joint_a, joint_b) in enumerate(limbs):
		for frame in range(tensor.shape[0]):
			starting_leg_lengths[frame, index] = torch.dist(initial_tensor[frame, joint_a], initial_tensor[frame, joint_b])

	initial_averages_limb_lengths = torch.mean(starting_leg_lengths, dim=0)


	averages_limb_lengths = initial_averages_limb_lengths.clone()

	averages_limb_lengths.unsqueeze_(0)
	averages_limb_lengths = averages_limb_lengths.expand(initial_tensor.shape[0],len(limbs))



	window = np.bartlett(args.args.direction_change_threshold*2+1)

	print("starting inflection_points")
	print(inflection_points)


	for epoch in range(10000):
		
		delta_x = tensor[1:]-tensor[:-1]

		delta_delta_x = delta_x[1:]-delta_x[:-1]




		loss = 0


		# differences between leg lengths
		for index, (joint_a, joint_b) in enumerate(limbs):
			#the length of the current leg for each frame
			difference = tensor[:, joint_a]- tensor[:, joint_b]
			distance = (difference*difference)
			this_sum = torch.sum(distance, dim=1)
			sqrt = torch.sqrt(this_sum)

			# loss += 0.0000001*torch.dist(averages_limb_lengths[:, index],sqrt)
			# loss += 1e-3*torch.dist(sqrt[1:],sqrt[:-1])
			loss += 1e-1*torch.mean((sqrt[1:]-sqrt[:-1])**2)
		

		loss += torch.mean((delta_delta_x)**2)+torch.mean((delta_x)**2)


		left_feet = tensor[:, LAnkle]
		right_feet = tensor[:, RAnkle]
		current_differences = left_feet-right_feet

		#project the current differences onto the forward vector
		pelvi = (tensor[:, LHip]+tensor[:, RHip])/2

		hip_vector_norm = torch.norm((tensor[:, RHip] - tensor[:, LHip]), dim = 1)
		hip_vector_norm = torch.unsqueeze(hip_vector_norm, 1)
		hip_vector_norm = hip_vector_norm.expand(hip_vector_norm.shape[0],3)

		up_vector_norm = torch.norm((tensor[:, Head] - pelvi), dim = 1)
		up_vector_norm = torch.unsqueeze(up_vector_norm, 1)
		up_vector_norm = up_vector_norm.expand(up_vector_norm.shape[0],3)


		hip_vectors = (tensor[:, RHip] - tensor[:, LHip])/hip_vector_norm
		hip_vectors = hip_vectors.detach()
		# hip_vectors = torch.div((tensor[:, RHip] - tensor[:, LHip]),torch.norm((tensor[:, RHip] - tensor[:, LHip]), dim = 1))
		up_vectors = (tensor[:, Head] - pelvi)/up_vector_norm
		up_vectors = up_vectors.detach()

		# up_vectors = torch.div((tensor[:, Head] - pelvi), torch.norm((tensor[:, Head] - pelvi), dim = 1))
		forward_vectors = torch.cross(up_vectors, hip_vectors, dim = 1)

		#project current_differences onto forward_vectors
		current_differences = forward_vectors * torch.bmm(current_differences.view(-1, 1, 3), forward_vectors.view(-1, 3, 1)).view(-1, 1)
		current_differences = current_differences.detach()

		
		left_feet_projection = tensor[:, LAnkle] - hip_vectors * dot_product(tensor[:, LAnkle], hip_vectors)
		right_feet_projection = tensor[:, RAnkle] - hip_vectors * dot_product(tensor[:, RAnkle], hip_vectors)


		# print(left_feet_projection[:3])
		# print(hip_vectors[:3])
		# print(tensor[:, LAnkle][:3])
		# exit()


		left_vector = left_feet_projection-pelvi
		right_vector = right_feet_projection-pelvi

		left_vector_norm = torch.norm(left_vector, dim = 1)
		left_vector_norm = torch.unsqueeze(left_vector_norm, 1)
		left_vector_norm = left_vector_norm.expand(left_vector_norm.shape[0],3)

		right_vector_norm = torch.norm(right_vector, dim = 1)
		right_vector_norm = torch.unsqueeze(right_vector_norm, 1)
		right_vector_norm = right_vector_norm.expand(right_vector_norm.shape[0],3)

		left_vector = left_vector/left_vector_norm
		right_vector = right_vector/right_vector_norm

		#relative leg angles
		sin_rla = dot_product(torch.cross(left_vector, right_vector, dim = 1), hip_vectors)
		cos_rla = dot_product(left_vector, right_vector)
		tan_rla = sin_rla / cos_rla

		#relative left foot
		sin_rlf = dot_product(torch.cross(left_vector, -1*up_vectors, dim = 1), hip_vectors)
		cos_rlf = dot_product(left_vector, -1*up_vectors)
		tan_rlf = sin_rlf / cos_rlf

		#relative right foot
		sin_rrf = dot_product(torch.cross(right_vector, -1*up_vectors, dim = 1), hip_vectors)
		cos_rrf = dot_product(right_vector, -1*up_vectors)
		tan_rrf = sin_rrf / cos_rrf

		#these are the absolute angles, find which one is forward relative to the other and multiply that by -1
		
		fps = 60

		#60 frames per second
		phi = torch.atan(tan_rla) 
		delta_phi = (phi[1:]-phi[:-1])*fps
		delta_delta_phi = (delta_phi[1:]-delta_phi[:-1])*fps

		rlf = torch.atan(tan_rlf) 
		delta_rlf = (rlf[1:]-rlf[:-1])*fps
		delta_delta_rlf = (delta_rlf[1:]-delta_rlf[:-1])*fps

		rrf = torch.atan(tan_rrf) 
		delta_rrf = (rrf[1:]-rrf[:-1])*fps
		delta_delta_rrf = (delta_rrf[1:]-delta_rrf[:-1])*fps



		rlf_mask = torch.ones((array.shape[0], 1), dtype=torch.float64, requires_grad=False)

		for x in range(0, len(inflection_points)-1, 2):
			for index in range(inflection_points[x], inflection_points[x+1]):
				rlf_mask[index] = 0


		rrf_mask = torch.ones((array.shape[0], 1), dtype=torch.float64, requires_grad=False)-rlf_mask



		theta = rlf*rlf_mask+rrf*rrf_mask
		delta_theta = delta_rlf*rlf_mask[:-1]+delta_rrf*rrf_mask[:-1]
		delta_delta_theta = delta_delta_rlf*rlf_mask[:-2]+delta_delta_rrf*rrf_mask[:-2]

		zeros = torch.zeros(delta_delta_theta.shape, requires_grad=False)

		equation_one = delta_delta_theta - torch.sin(theta[:-2])

		# print(torch.sin(theta[:-2])[170:180])
		# print(delta_delta_theta[170:180])

		loss += 1e-9*torch.mean((equation_one-zeros)**2)

		equation_two = delta_delta_theta - delta_delta_phi + delta_theta[:-1]**2 * torch.sin(phi[:-2]) - torch.cos(theta[:-2]) * torch.sin(phi[:-2])

		# print(delta_delta_theta[170:180])
		# print(delta_delta_phi[170:180])
		# print((delta_theta[:-1]**2 * torch.sin(phi[:-2]))[170:180])
		# print((torch.cos(theta[:-2])** torch.sin(phi[:-2]))[170:180])
		# exit()

		loss += 1e-9*torch.mean((equation_two-zeros)**2)


		transition_mask = torch.zeros((array.shape[0], 1), requires_grad=False)

		for frame in inflection_points:
			transition_mask[frame-1] = 1

		new_theta = theta[:-1]
		new_delta_theta = torch.cos(2*theta[:-2])*delta_theta[:-1]
		new_phi = 2*theta[:-1]
		new_delta_phi = torch.cos(2*theta[:-2])*(1-torch.cos(2*theta[:-2]))*delta_theta[:-1]


		loss += 1e-5*torch.mean(((theta[1:]+new_theta)*transition_mask[:-1])**2)
		loss += 1e-5*torch.mean(((delta_theta[1:]-new_delta_theta)*transition_mask[:-2])**2)
		loss += 1e-5*torch.mean(((phi[1:]+new_phi)*transition_mask[:-1])**2)
		loss += 1e-5*torch.mean(((delta_phi[1:]-new_delta_phi)*transition_mask[:-2])**2)


		if(epoch%100==0):
			print(epoch)
			print(loss)
		
		loss.backward()

		with torch.no_grad():
			tensor -= 100 * tensor.grad
			tensor.grad.zero_()


			for point in inflection_points:
				tensor[point] = initial_tensor[point]


	ending_leg_lengths = torch.ones([tensor.shape[0], len(limbs)], dtype=torch.float64, requires_grad=False)

	#differences between leg lengths
	for index, (joint_a, joint_b) in enumerate(limbs):
		for frame in range(tensor.shape[0]):
			ending_leg_lengths[frame, index] = torch.dist(tensor[frame, joint_a], tensor[frame, joint_b])

	averages_limb_lengths = torch.mean(starting_leg_lengths, dim=0)
	print("averages_limb_lengths after")
	print(averages_limb_lengths)



	return_array = tensor.detach().numpy()


	inflection_points = find_frames_changing_direction(return_array, smooth = False, file_name = file_name)

	print()

	return return_array