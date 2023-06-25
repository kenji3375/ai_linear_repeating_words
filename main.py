
import string_data as data

import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt

import math

# Check PyTorch version
print(torch.__version__)




#make data
X = []
Y = []




for item in data.data:
    X.append(item[0])
    Y.append(item[1])
#print(x)
#print(y)

#make X tensor

#print(X[:10])

X = torch.tensor(X)


#make Y tensor

#for i in range(len(Y)):     #make each item in Y a list
    #Y[i] = [Y[i]]
Y = torch.tensor(Y)

# a list is a 1d tensor

print("===========")
#print(X[:10])
print("===========")
#print(Y[:10])


#split data

X_train = X[:int(len(X)*0.8)]
X_test = X[int(len(X)*0.8):]

Y_train = Y[:int(len(Y)*0.8)]
Y_test = Y[int(len(Y)*0.8):]

print("===========")
print( len (X_train))
print( len (X_test))
print( len (Y_train))
print( len (Y_test))


print(X_train[:10])


X_train = X_train.float()
X_test = X_test.float()

Y_train = Y_train.float()
Y_test = Y_test.float()



#make model

#the model will be a simple linear regression model

model = nn.Linear(data.max_len, data.max_len)

'''
model = nn.Sequential(
    nn.Linear(data.max_len, data.max_len),
    #nn.ReLU()
)
'''

answer = input("Do you want to load a model? (y/n) ")

if answer == "y":
    file_path = "model.pt"
    model.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))

answer = input("Do you want to train the model? (y/n) ")

train = False

if answer == "y":
    train = True

print(model)

#loss function adam

loss_function = nn.MSELoss()

#optimizer


epochs = data.number_of_strings
_lr = float(1/(epochs))*10

optimizer = torch.optim.Adam(model.parameters(), lr=_lr)

#train



# RuntimeError: Found dtype Long but expected Float

X_train = X_train.float()
X_test = X_test.float()

Y_train = Y_train.float()
Y_test = Y_test.float()

if train == False:
    epochs = 0


for epoch in range(epochs):

    print("epoch: " + str(epoch) + " of " + str(epochs) + " epochs")

    Y_pred = model(X_train)
    loss = loss_function(Y_pred, Y_train)
    print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    #test

    Y_pred = model(X_test)
    loss = loss_function(Y_pred, Y_test)
    print('Test loss: {}'.format(loss.item()))

#save model

#torch.save(model.state_dict(), 'model.pth')



def custom_round(number):
    decimal_part = number - math.floor(number)
    if decimal_part < 0.5:
        return math.floor(number)
    else:
        return math.ceil(number)

number = 0.7291666666666666
rounded_number = custom_round(number)
print(rounded_number)  # Output: 3





while True:
    print("===========")

    string = input("Enter string: ")

    test = False

    if string == "save":
        name = input("Enter the name: ")
        torch.save(model.state_dict(), name+".pt")
        print("model saved")
        continue
    if string == "test":

        errors = 0

        samples = 9999

        if len(data.data) < samples:
            samples = len(data.data)

        for i in range(samples):
            
            string = data.data[i][0]

            string2 = ""

            for char in string:
                string2 += data.letter_tokens[int(char)]

            string = string2

            start_string = string
            
            #print(string)

            tok_arr = []

            for char in string:
                for i in range(0, len(data.letter_tokens)):
                    if  char == data.letter_tokens[i] and len(tok_arr) < data.max_len:
                        tok_arr.append((i))


            while len(tok_arr) < data.max_len:
                tok_arr.append(0)
            


            #print(tok_arr)
            #print(len(tok_arr))

            #print("===========")

            prediction = model(torch.tensor(tok_arr).float())

            prediction = prediction.tolist()


            prediction2 = []

            for element in prediction:
                prediction2.append(element)
            
            prediction = prediction2

            prediction2 = []

            for element in prediction:
                prediction2.append(custom_round(element))
            
            #print(prediction2)

            prediction = prediction2
            
            string = ""
            for element in prediction:
                if element < 0:
                    element = 0
                string += data.letter_tokens[int(element)]
            if start_string != string:
                errors += 1
                #print(string)
        print(str(errors) + "/" + str(samples) + " errors")
        test = True

    #make string max_len long


    #make string into tensor using tokens


    tok_arr = []

    for char in string:
        for i in range(0, len(data.letter_tokens)):
            if  char == data.letter_tokens[i] and len(tok_arr) < data.max_len:
                tok_arr.append((i))


    while len(tok_arr) < data.max_len:
        tok_arr.append(0)
    


    print(tok_arr)
    print(len(tok_arr))

    print("===========")

    prediction = model(torch.tensor(tok_arr).float())

    prediction = prediction.tolist()

    prediction2 = []

    for element in prediction:
        prediction2.append(element)
    
    prediction = prediction2

    prediction2 = []

    for element in prediction:
        prediction2.append(custom_round(element))
    
    print(prediction2)

    prediction = prediction2
    
    string = ""
    for element in prediction:
        if element < 0:
            element = 0
        string += data.letter_tokens[int(element)]

    #print(prediction[0])
    if test == False:
        print("prediction: " + string) #str(prediction))
        print("===========")
