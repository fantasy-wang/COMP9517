import torch
import pandas as pd

# Load data(do not change)
data = pd.read_csv("src/mnist_train.csv")
train_data = data[:2000]
test_data = data[2000:2500]

print(train_data)


'''
# ----- Prepare Data ----- #
# step one: preparing your data including data normalization

# step two: transform np array to pytorch tensor

# ----- Build CNN Network ----- #
# Define your model here
class mymodel(torch.nn.Module):
    def __init__(self):
        super(mymodel, self).__init__()
        pass

    def forward(self, x):
        pass

# Define our model
model = None
# Define your learning rate
learning_rate = None
# Define your optimizer
optimizer = None
# Define your loss function
criterion = None

# ----- Complete PlotLearningCurve function ----- #
def PlotLearningCurve(epoch, trainingloss, testingloss):
    pass

# ----- Main Function ----- #
trainingloss = []
testingloss = []
# Define number of iterations
epochs = None
for epoch in range(1, epochs + 1):
    model.train()
    # step one : fit your model by using training data and get predict label
    
    # step two: calculate your training loss
    loss = None
    # step three: calculate backpropagation
    
    # step four: update parameters
    
    # step five: reset our optimizer

    # step six: store your training loss
    trainingloss += loss.item(),
    # step seven: evaluation your model by using testing data and get the accuracy
    with torch.no_grad():
        model.eval()
        # predict testing data
        
        # calculate your testing loss
        loss = pass
        # store your testing loss
        testingloss += loss.item(),
        if epoch % 10 == 0:
            # get labels with max values

            # calculate the accuracy
            acc = None
            print('Epoch:', epoch, 'Test Accuracy:', acc)
'''