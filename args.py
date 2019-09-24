import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--height', type=float, default = 185)		#height in cm
parser.add_argument('--draw_plot', type=int, default = 0)		#1 is true, 0 is false



args = parser.parse_args()
