from PIL import Image
from nueral_network import Neural_netwokr
from readfile import Readfile
import numpy as np
import os
from visualization import vi_run
from chart import Chart

OUTPUT = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        , [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        , [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        , [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        , [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        , [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        , [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        , [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        , [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        , [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
HIDDEN_NODE = 25
INPUT_NODE = 784
OUTPUT_NODE = 10
EPOCH = [20000, 500] #gd-> 20000, sgd->500


if __name__ == "__main__":
	readfile = Readfile()
	training_input = np.array(readfile.get_training_input())
	training_output = np.array(readfile.get_training_output())

	ans = input(
		"algorigthm of optimization? [1.gradient decsent 2.stochastic gradient decsent]")
	ans2 = input(
		"Activation function? [1.linear 2.sigmoid]")
	vi_run(HIDDEN_NODE, INPUT_NODE, OUTPUT_NODE) #VISUALIZATION
	neural = Neural_netwokr(HIDDEN_NODE, ans2)
	for i in range(EPOCH[ans-1]):
		neural.train(training_input, training_output, ans, ans2)
	if ans == 1:
		chart = Chart(EPOCH[ans-1], neural.get_chartYA())
	neural.predict(training_input, training_output, len(training_input))


# ##################################################
# 	dinput = []
# 	doutput = []
# 	for filename in os.listdir("x"):
# 			# print "filename: ", filename, "i: ", i
# 		im = Image.open("x/" + filename)
# 		pix = im.load()
# 		temp = []
# 		for m in range(0, im.size[0]):
# 			for k in range(0, im.size[1]):
# 				temp.append(pix[m, k])
# 		dinput.append(temp)
# 		doutput.append(OUTPUT[0])
# ###################################################

