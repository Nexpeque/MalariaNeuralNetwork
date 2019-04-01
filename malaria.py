import matplotlib.pyplot as plt
import tensorflow as tf 
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split 


numberOfParameters = 10 #This number represents the amount of columns inside the dataset, in this case 10 values of inputs and 1 lablel

def read_dataset():
    #this function returns the values of X and Y both  Matrix and vector respectivly
    df = pd.read_csv("dataset_test.csv", sep=';')
    print(df)
    x = df[df.columns[0:numberOfParameters]].values
    y = df[df.columns[numberOfParameters]]  #This vector has the value of whether a specific person has or not malaria

    encoder = LabelEncoder()
    encoder.fit(y) 
    y = encoder.transform(y)
    with tf.Session() as sesh:
        y = sesh.run(tf.one_hot(y, len(np.unique(y)))) #This type of encoding makes only one input active
    
    return (x,y) 

#Read the dataset 
X, Y =  read_dataset();

#Shuffle the dataset to mix the rows randomly
X, Y = shuffle(X, Y, random_state=1) 

train_x, test_x, train_y, test_y = train_test_split(X,Y,test_size=0.20, random_state=412)   

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)

learning_rate = 0.025 
epochs = 1000 #Number of iterations that will be done in order to minimize the error
cost = np.empty(shape=[1], dtype=float)
n_dim = X.shape[1] #The shape of the matrix ?x?
n_class = 2 #This value sets the number of classes inside the model (Severe malaria, non severe malaria)
model_path="NMI" #Path where the model of the neural network will be stored

n_hidden_1 = 8
n_hidden_2 = 8
n_hidden_3 = 8
n_hidden_4 = 8

x = tf.placeholder(tf.float32, [None, n_dim])
W = tf.Variable(0.1*tf.zeros([n_dim, n_class]))
b = tf.Variable(0.1*tf.zeros([n_class]))
y_ = tf.placeholder(tf.float32, [None, n_class])

def multilayer_perceptron(x, weights, biases):
    #Makes the multiplication of x*W*b
        #runs the layer inside the sigmoid function   
    layer_1 =  tf.add(tf.matmul(x, weights['h1']),biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)

    layer_2 =  tf.add(tf.matmul(layer_1, weights['h2']),biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)

    layer_3 =  tf.add(tf.matmul(layer_2, weights['h3']),biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)
        #runs the layer in the relu function
    layer_4 =  tf.add(tf.matmul(layer_3, weights['h4']),biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']

    return out_layer
    
weights = {
    'h1': tf.Variable(tf.truncated_normal([n_dim,n_hidden_1])),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1,n_hidden_2])),
    'h3': tf.Variable(tf.truncated_normal([n_hidden_2,n_hidden_3])),
    'h4': tf.Variable(tf.truncated_normal([n_hidden_3,n_hidden_3])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_4,n_class]))
}
biases = {
    'b1' : tf.cast(tf.Variable(tf.truncated_normal([n_hidden_1])),tf.float32),
    'b2' : tf.cast(tf.Variable(tf.truncated_normal([n_hidden_2])),tf.float32), 
    'b3' : tf.cast(tf.Variable(tf.truncated_normal([n_hidden_3])),tf.float32), 
    'b4' : tf.cast(tf.Variable(tf.truncated_normal([n_hidden_4])),tf.float32), 
    'out' : tf.cast(tf.Variable(tf.truncated_normal([n_class])),tf.float32)
}

init = tf.global_variables_initializer()

saver = tf.train.Saver()

#Call the model defined
#Send the values, the wigths and biases to the neural network
#Returns the out layer
y = multilayer_perceptron(x,weights,biases)


cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y, labels = y_))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

with tf.Session() as sesh:
    sesh.run(init)
    mse_history = []
    accuracy_history = []
    cost_history = []

    for epoch in range(epochs):
        sesh.run(training_step, feed_dict={x:train_x, y_:train_y})
        cost = sesh.run(cost_function, feed_dict={x:train_x, y_:train_y})
        cost_history = np.append(cost_history, cost)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        pred_y = sesh.run(y, feed_dict={x: test_x})
        mse = tf.reduce_mean(tf.square(pred_y - test_y))   
        mse_ = sesh.run(mse)
        mse_history.append(mse_)
        accuracy = (sesh.run(accuracy, feed_dict={x: train_x, y_:train_y}) )
        accuracy_history.append(accuracy)

        print('epoch: ', epoch, '-', 'cost: ',cost,'- MSE',mse_,'- Train Accuracy: ', accuracy )

    save_path = saver.save(sesh, model_path)
    print("Model saved in file: %s"% save_path)

    plt.plot(mse_history, 'r')
    plt.show()
    plt.plot(accuracy_history)
    plt.show()

    #print the final accuracy
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Test Accuracy", (sesh.run(accuracy, feed_dict={x: test_x, y_: test_y})))

    #print the final mean squared error
    pred_y = sesh.run(y, feed_dict={x: test_x})
    mse = tf.reduce_mean(tf.square(pred_y - test_y))
    print("MSE: %.4f" %sesh.run(mse))
    