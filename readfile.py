import os
from PIL import Image

PATH = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
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
NUMBER_OF_LETTER = 10

class Readfile:

    def __init__(self):
        self.input = []
        self.output = []
        for i in range(NUMBER_OF_LETTER):
            for filename in os.listdir(PATH[i]):
                im = Image.open(PATH[i] + '/' + filename)
                pix = im.load()
                temp = []
                for m in range(0, im.size[0]):
                    for k in range(0, im.size[1]):
                        temp.append(pix[m, k])
                self.input.append(temp)
                self.output.append(OUTPUT[i])


    def get_training_input(self):
        return self.input


    def get_training_output(self):
        return self.output
