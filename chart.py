import matplotlib.pyplot as plt

class Chart:

    def __init__(self, iteraion_n, lossA):
        self.x = []
        self.yA = []
        # self.yB = []
        for i in range(iteraion_n):
            self.x.append(i)
        for i in range(iteraion_n):
            self.yA.append(lossA[i])
            # self.yB.append(lossB[i])

        plt.plot(self.x, self.yA, color='g')
        # plt.plot(self.x, self.yB, color='orange')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Gradient descent')
        plt.show()
