import matplotlib.pyplot as plt
import numpy as np

from acquisition_functions.lower_confidence_bound import LowerConfidenceBound
from bayesian_optimisation import BayesianOptimisation
from kernels.gaussian_kernel import GaussianKernel

from objective_functions.six_hump_camel import SixHumpCamelObjectiveFunction
from objective_functions.univariate_objective_function import UnivariateObjectiveFunction

from gaussian_process import GaussianProcess
from kernels.gaussian_kernel import GaussianKernel
from kernels.matern_kernel import MaternKernel

objective_function = UnivariateObjectiveFunction(additional_gaussian_noise_std=0.5)

kernel_gaussian = GaussianKernel(-1., 0., -1.)
gaussian_process = GaussianProcess(kernel_gaussian)

gaussian_process.plot_with_samples(5, objective_function)
plt.show()

boundaries, = objective_function.boundaries
x = np.linspace(*boundaries, 50).reshape((-1, 1))
y = objective_function.evaluate(x).reshape((-1, 1))
gaussian_process.initialise_dataset(x, y)

gaussian_process.plot_with_samples(5, objective_function)
plt.show()
