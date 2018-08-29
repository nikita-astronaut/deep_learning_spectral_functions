import numpy as np

def exponential_kernel(omega, tau):
	return np.exp(-omega * tau)

def thermal_kernel(omega, tau):
	return np.cosh(omega * (tau - 0.5)) / np.sinh(omega / 2.0)
