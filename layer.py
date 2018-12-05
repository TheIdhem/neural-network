import numpy as np
import numpy


class Layer:

    def cal_output(self, weight, linear, input):
        sum = np.dot(input, weight)
        result = 0
        if linear == 1:
            result = self.linear(sum)
        else:
            result = self.sigmoid(sum)
        self.output = result
        return result


    def get_output(self):
        return self.output


    def sigmoid(self, x):
        return 1.0 / (1 + numpy.exp(-x))


    def linear(self, x): # y = x
        return x