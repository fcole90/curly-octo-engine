import unittest
import numpy as np

import math_fun.differentiable_argmax as mf


class DifferentiableArgmaxTester(unittest.TestCase):
    def test_np_max_amplifier_array(self):
        for i in range(1000):
            test_matrix = np.random.normal(scale=i, size=(5, 100))
            test_matrix_amplified = mf.np_max_amplifier(test_matrix, axis=1)

            # print("Test", i)
            # print("test matrix\n",test_matrix)
            # print("---------")
            # print(np.amax(test_matrix, axis=1))
            # print("---------")
            # print(np.divide(np.transpose(test_matrix),
            #                 np.add(np.amax(test_matrix, axis=1), 0.1e-100)))
            # print("amplified\n",test_matrix_amplified)

            # self.assertLessEqual(np.max(test_matrix_amplified), 1.0)
            # self.assertGreaterEqual(np.min(test_matrix_amplified), 0.0)

    def test_np_differentiable_argmax(self):
        """
        Checks that the root mean squared error is always below 1.

        This check compares differentiable argmax with np.argmax.

        Returns
        -------

        """
        for i in range(1000):
            test_matrix = np.random.normal(scale=i, size=(100, 5))
            #test_matrix = np.abs(np.random.normal(scale=i, size=(3, 3)))

            test_axis = 1
            amplification = 1000
            test_matrix_amplified = mf.np_max_amplifier(test_matrix, axis=test_axis, amp=amplification)
            test_argmax = mf.np_differentiable_argmax(test_matrix, axis=test_axis, power=amplification)
            true_argmax = np.argmax(test_matrix, test_axis)

            # print("Test", i)
            # print("test matrix\n",test_matrix)
            # print("---------")
            # print(np.amax(test_matrix, test_axis))
            # print("---------")
            # print("divided\n",np.transpose(np.divide(np.transpose(test_matrix),
            #                 np.add(np.amax(test_matrix, axis=test_axis), 0.1e-100))))
            # print("amplified\n",test_matrix_amplified)
            # print("true argmax", np.round(true_argmax), "values:", test_matrix[:][true_argmax])
            # print("test argmax", np.round(test_argmax), "values:", test_matrix[:][test_argmax.astype(int)])

            rmse = np.sqrt(np.nanmean(np.square(np.subtract(test_argmax, true_argmax))))
            self.assertLessEqual(rmse, 1)


if __name__ == '__main__':
    unittest.main()
