import keras
from keras.models import Sequential
from keras.layers import Dense
from ann_visualizer.visualize import ann_viz

# HIDDEN_LEN = 20
# INPUT_LEN = 784
# OUTPUT_LEN = 10


def vi_run(HIDDEN_LEN, INPUT_LEN, OUTPUT_LEN):
    network = Sequential()
    #Hidden Layer#1
    network.add(Dense(units=HIDDEN_LEN,
                    activation='sigmoid',
                    kernel_initializer='uniform',
                    input_dim=INPUT_LEN))  # 25 * 784 + 25(bias)

    #output Layer#2
    network.add(Dense(units=OUTPUT_LEN,
                    activation='sigmoid',
                    kernel_initializer='uniform'))  #para 10 * 25 + 10(bias)

    ann_viz(network, title="network")
    print(network.summary())
