import numpy as np

def uniform_data_generator(omegas):
	spectral_function = np.random.uniform(low=0, high=1, size=len(omegas))
	return spectral_function / np.sum(spectral_function)

def correlator_generator(spectral_function, kernel, omegas, taus):
	correlator = []
	for tau in taus:
		corr = sum([spectral_function[idx] * kernel(omega, tau) for idx, omega in enumerate(omegas)])
		correlator.append(corr)
	return np.array(correlator)

def data_generator(spectral_generator, kernel, omegas, taus, batch_size):
	while True:
		X = np.array([spectral_generator(omegas) for _ in range(batch_size)])
		y = np.array([correlator_generator(x, kernel, omegas, taus) for x in X])
		return X, y