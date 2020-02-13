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

boundaries, = objective_function.boundaries
x_train = np.linspace(*boundaries, 50).reshape((-1, 1))
y_train = objective_function.evaluate(x_train).reshape((-1, 1))

x_test = np.linspace(*boundaries, 150).reshape((-1, 1))
y_test = objective_function.evaluate(x_test).reshape((-1, 1))


gaussian_process.initialise_dataset(x_train, y_train)
print(f"LPD Gaussian Process without marginal likelihood optimisation: {gaussian_process.get_log_predictive_density(x_test, y_test)}")
gaussian_process.plot(objective_function)

print('-' * 50)
gaussian_process.optimise_parameters(disp=False)

print(f"LPD optimised Gaussian Process: {gaussian_process.get_log_predictive_density(x_test, y_test)}")
gaussian_process.plot(objective_function)
