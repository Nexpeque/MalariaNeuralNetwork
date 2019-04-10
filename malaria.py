import matplotlib.pyplot as plt
import tensorflow as tf 
import numpy as np
import pandas as pd



numberOfParameters = 10 #This number represents the amount of columns inside the dataset, in this case 10 values of inputs and 1 lablel

def read_dataset():
    dir_path = ""
    df = pd.read_csv("dataset_test.csv", sep=';')
    X = df[df.columns[0:10]].values
    y = df[df.columns[10]]
    
    return (X,Y)


def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels),labels] = 1
    return one_hot_encode

# Read the dataset
X, Y = read_dataset()

# Shuffle the dataset to mix up the rows


print("train_x.shape",train_x.shape)
print("train_y.shape",train_y.shape)
print("test_x.shape",test_x.shape)
print("test_y.shape",test_y.shape)

learning_rate = 0.025 
epochs = 1000 #Number of iterations that will be done in order to minimize the error
training_epochs = 1000
cost_history = np.empty(shape=[1], dtype=float)
# Number of features <=> number of columns
n_dim = X.shape[1] 
print("n_dim",n_dim)
n_class = 2 #This value sets the number of classes inside the model (Severe malaria, non severe malaria)
model_path="NMI" #Path where the model of the neural network will be stored

n_hidden_1 = 16

x = tf.placeholder(tf.float32,[None, n_dim])
y_ = tf.placeholder(tf.float32,[None, n_class])

# Model parameters
W = tf.Variable(tf.zeros([n_dim,n_class]))
b = tf.Variable(tf.zeros([n_class]))


#Define the model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activations
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

weights = {
    'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_1, n_class])),
    }

biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
    'out': tf.Variable(tf.truncated_normal([n_class])),
    }


init = tf.global_variables_initializer()

saver = tf.train.Saver()

#Call the model defined
#Send the values, the wigths and biases to the neural network
#Returns the out layer
y = multilayer_perceptron(x, weights, biases)


cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = y, labels = y_))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

with tf.Session() as sesh:
    sesh.run(init)

    mse_history = []
    accuracy_history = []

    for epoch in range(training_epochs):
        sesh.run(training_step, feed_dict={x:train_x, y_:train_y})
        cost = sesh.run(cost_function,feed_dict={x:train_x, y_:train_y})
        cost_history = np.append(cost_history, cost)
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #     print("Accuracy: ", (sesh.run(accuracy, feed_dict={x:test_x, y_:test_y})))
        pred_y = sesh.run(y,feed_dict={x:test_x} )
        mse = tf.reduce_mean(tf.square(pred_y - test_y))
        mse_ = sesh.run(mse)
        mse_history.append(mse_)
        accuracy = (sesh.run(accuracy,feed_dict={x:train_x, y_:train_y}))
        accuracy_history.append(accuracy)
        print('epoch: ', epoch,' - ', 'cost: ', cost, " - MSE: ", mse_, "- Train Accuracy: ", accuracy)

    save_path = saver.save(sesh, model_path)
    print("Model saved in file: %s", save_path)

    plt.plot(accuracy_history)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

    plt.plot(range(len(cost_history)),cost_history)
    plt.axis([0,training_epochs,0,np.max(cost_history)/100])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    # Print the final mean square error

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(tf.square(pred_y - test_y),tf.float32))
    print("Test Accuracy: ", (sesh.run(accuracy, feed_dict={x:test_x, y_:test_y} )))

    # Print the final mean square error
    pred_y = sesh.run(y, feed_dict={x:test_x})  
    mse = tf.reduce_mean(tf.square(pred_y- test_y))
    print("MSE: %.4f" % sesh.run(mse))
    