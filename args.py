import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--height', type=float, default = 185)		#height in cm
parser.add_argument('--draw_plot', type=int, default = 0)		#1 is true, 0 is false

#	min number of frames allowed for a threshold of min frames seperating change in direction
#	meant to reduce noise
parser.add_argument('--direction_change_threshold', type=int, default = 10)		




args = parser.parse_args()
