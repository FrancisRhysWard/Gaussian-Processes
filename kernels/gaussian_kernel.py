import numpy as np

from kernels.abstract_kernel import Kernel


class GaussianKernel(Kernel):
    def __init__(self,
                 log_amplitude: float,
                 log_length_scale: float,
                 log_noise_scale: float,
                 ):
        super(GaussianKernel, self).__init__(log_amplitude,
                                             log_length_scale,
                                             log_noise_scale,
                                             )

    def get_covariance_matrix(self,
                              X: np.ndarray,
                              Y: np.ndarray,
                              ) -> np.ndarray:
        """
        :param X: numpy array of size n_1 x l for which each row (x_i) is a data point at which the objective function can be evaluated
        :param Y: numpy array of size n_2 x m for which each row (y_j) is a data point at which the objective function can be evaluated
        :return: numpy array of size n_1 x n_2 for which the value at position (i, j) corresponds to the value of
        k(x_i, y_j), where k represents the kernel used.
        """

        s2 = np.exp(self.log_amplitude * 2)

        l = np.exp(self.log_length_scale) ** 2

        K = np.zeros((X.shape[0], Y.shape[0]))

        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                K[i,j] = np.linalg.norm(x-y) ** 2

        return s2 * np.exp((-1 /(2*l)) * K )

    def __call__(self,
                 X: np.ndarray,
                 Y: np.ndarray,
                 ) -> np.ndarray:
        return self.get_covariance_matrix(X, Y)
