from typing import Union

import numpy as np
from scipy.stats import norm

from acquisition_functions.abstract_acquisition_function import AcquisitionFunction
from gaussian_process import GaussianProcess


class ExpectedImprovement(AcquisitionFunction):
    def _evaluate(self,
                  gaussian_process: GaussianProcess,
                  data_points: np.ndarray
                  ) -> np.ndarray:
        """
        Evaluates the acquisition function at all the data points
        :param gaussian_process:
        :param data_points: numpy array of dimension n x m where n is the number of elements to evaluate
        and m is the number of variables used to calculate the objective function
        :return: a numpy array of shape n x 1 (or a float) representing the estimation of the acquisition function at
        each point
        """

        mean, std = gaussian_process.get_gp_mean_std(data_points)
        n, l = mean.shape

        f_best = np.min(mean)
        gamma = (f_best - mean)/std

        cdf = norm.cdf(gamma)

        expected_improvement = std*(gamma*cdf + np.random.normal(size=n))

        return expected_improvement
        # TODO
