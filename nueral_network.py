import random
import numpy as np
import numpy
from layer import Layer

LEARNING_RATE = [0.0008, 0.03]  # gd->0.001, std->0.03
LAMBDA = 0.00001
ONE = 1
PATH = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
NUMBER_OF_LETTER = 10


class Neural_netwokr:
    def __init__(self, hidden_len, ans2, input_len=784, output_len=10):
        self.input_len = input_len
        self.input_layer_neurons = []
        self.hidden_len = hidden_len
        self.hidden_layer_neurons = []
        self.output_len = output_len
        self.output_layer_neurons = []
        self.activation_fun = ans2  # 1.linear 2.sigmoid

        self.w1 = np.random.randn(self.input_len, self.hidden_len)
        self.w2 = np.random.randn(self.hidden_len, self.output_len) ## hidden_len*10
        self.b1 = np.random.randn(1, hidden_len)
        self.b2 = np.random.randn(1, output_len)

        self.chartYA = []
        # self.chartYB = []


    def get_chartYA(self):
        return self.chartYA

    def get_chartYB(self):
        return self.chartYB


    def train(self, rgba):
        self.forward(rgba)
        self.backward_propagation()


    def s_forward(self, rgba, ans2):
        o = self.hidden_layer.cal_output(self.w1, ans2, rgba)
        oo = self.hidden_layer.cal_output(self.w2, ans2, o)
        return oo, o


    def linear(self, x):
        return x


    def linearPrime(self, x):
        return 1.

    def sigmoidPrime(self, x):
        return x * (1 - x)


    def sigmoid(self, x):
        return 1.0 / (1 + numpy.exp(-x))
        # return 1/(1+np.exp(-x))
        # return 1/(1+exp(-x))


    def make_np(self, array):
        result_tm = []
        for i in array:
            temp = []
            temp.append(i)
            result_tm.append(temp)
        result = np.array(result_tm)
        return result


    def s_backward_propagation(self, oo, training_output, training_input, oh, is_a):
        output_loss = training_output - oo # avali trainging az ax
        # self.chartY.append(output_loss[0]) #for chart
        output_delta = output_loss * self.sigmoidPrime(oo)
        r_output_delta = self.make_np(output_delta)
        r_oh = self.make_np(oh)
        self.w2 += (LEARNING_RATE[1] * np.dot(r_oh, r_output_delta.T))

        hidden_loss = np.dot(output_delta, self.w2.T)
        hidden_delta = hidden_loss * self.sigmoidPrime(oh)
        r_hidden_delta = self.make_np(hidden_delta)
        r_training_input = self.make_np(training_input)
        self.w1 += (LEARNING_RATE[1] * np.dot(r_training_input, r_hidden_delta.T))


    def train(self, training_input, training_output, ans, ans2):
        if ans == 1:
            self.g_forward(training_input)
            self.g_backward_propagation(training_output, training_input)
        elif ans == 2:
            self.hidden_layer = Layer()
            self.output_layer = Layer()
            for j in range(len(training_input)):
                oo, oh = self.s_forward(training_input[j], ans2)
                if j < 50:
                    self.s_backward_propagation(oo, training_output[j], training_input[j], oh, True)
                else:
                    self.s_backward_propagation(oo, training_output[j], training_input[j], oh, False)


    def predict(self, input, training_output, input_len):
        counter = 0
        for i in range(len(input)):
            in1 = self.make_np(input[i])
            sum = np.dot(in1.T, self.w1)
            hio = self.sigmoid(sum)  # 490*(len_hidden)4
            sum_ = np.dot(hio, self.w2)
            out = self.sigmoid(sum_)
            c = out[0]
            max = -1
            ind = -1
            for j in range(len(c)):
                if c[j] > max:
                    max = c[j]
                    ind = j
            if PATH[ind] == PATH[i/(input_len/NUMBER_OF_LETTER)]:
                counter += 1
        print "accuracy: ", (counter*100)/input_len, "% ,counter: ", counter


    def g_forward(self, training_input):
        self.cal_hidden_layer(training_input)
        self.cal_output_layer()


    def cal_hidden_layer(self, training_input): #sum size: 490 * hidden_len
        sum = np.dot(training_input, self.w1) + self.b1
        if self.activation_fun == 1:
            self.hidden_layer_outputs = self.linear(sum)
        elif self.activation_fun == 2:
            self.hidden_layer_outputs = self.sigmoid(sum) ##490*(len_hidden)4


    def cal_output_layer(self):
        sum = np.dot(self.hidden_layer_outputs, self.w2) + self.b2
        if self.activation_fun == 1:
            self.output_layer_outputs = self.linear(sum)
        elif self.activation_fun == 2:
            self.output_layer_outputs = self.sigmoid(sum) ##490*(len_hidden)4


    def g_backward_propagation(self, training_output, training_input):
        output_loss = training_output - self.output_layer_outputs # avali trainging az ax
        self.chartYA.append(output_loss[0][0])  # for chart
        # self.chartYB.append(output_loss[50][1])  # for chart
        if self.activation_fun == 1:
            output_delta = output_loss * self.linearPrime(self.output_layer_outputs)
        elif self.activation_fun == 2:
            output_delta = output_loss * self.sigmoidPrime(
                    self.output_layer_outputs)  # 490*(len_hidden)4

        self.w2 += (LEARNING_RATE[0] * np.dot(self.hidden_layer_outputs.T, output_delta))
        self.w2 -= (LAMBDA * self.w2) #for l2 regularization 
        self.b2 += (LEARNING_RATE[0] * np.sum(output_delta, axis=0))

        hidden_loss = np.dot(output_delta, self.w2.T)
        if self.activation_fun == 1:
            hidden_delta = hidden_loss * self.linearPrime(self.hidden_layer_outputs)
        elif self.activation_fun == 2:
            hidden_delta = hidden_loss * self.sigmoidPrime(
                    self.hidden_layer_outputs)  # 490*(len_hidden)4

        self.w1 += (LEARNING_RATE[0] * np.dot(training_input.T, hidden_delta))
        self.w1 -= (LAMBDA * self.w1) #for l2 regularization 
        self.b1 += (LEARNING_RATE[0] * np.sum(hidden_delta, axis=0))
