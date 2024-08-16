#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 21:21:50 2023

@author: meghanapuli
"""

'''
Problem statement

Restaurent profit prediction. 

Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new outlet.
You would like to expand your business to cities that may give your restaurant higher profits. The chain already 
has restaurants in various cities and you have data for profits and populations from the cities. You also have data 
on cities that are candidates for a new restaurant. For these cities, you have the city population.
Can you use the data to help you identify which cities may potentially give your business higher profits?
'''

import numpy as np
import matplotlib.pyplot as plt

def load_data():
    data = np.loadtxt("population.txt", delimiter=',')
    X = data[:,0]
    y = data[:,1]
    return X, y

# load the dataset
X_train, y_train = load_data()

# x_train represents the city population times 10,000
# For example, 6.1101 means that the population for that city is 61,101
print("Type of X_train:",type(X_train))
print("First five elements of X_train are:\n", X_train[:5]) 

# y_train represents restaurant's average monthly profits in each city, in units of $10,000.
print("\nType of y_train:",type(y_train))
print("First five elements of y_train are:\n", y_train[:5]) 

print ('\nThe shape of x_train is:', X_train.shape)
print ('The shape of y_train is: ', y_train.shape)
print ('Number of training examples (m):', len(X_train))

# Create a scatter plot of the data. 
plt.scatter(X_train, y_train, marker='x', c='r') 

# Set the title
plt.title("Profits vs. Population per city")
# Set the y-axis label
plt.ylabel('Profit in $10,000')
# Set the x-axis label
plt.xlabel('Population of City in 10,000s')
plt.show()

# list to store the cost 
cost_history = []
num_iterations = []

# compute the prediction of the model
def compute_model_output(X, w, b):

    f_wb = w * X + b
        
    return f_wb

def compute_cost(X, y, w, b): 

    # number of training examples
    m = X.shape[0] 
    total_cost = 0

    for i in range(m):
        f_wb = w*X[i] + b
        cost = (f_wb - y[i])**2
        total_cost += cost
        
    total_cost = 1 / (2 * m) * total_cost
    
    return total_cost

def compute_gradient(X, y, w, b): 
    
    # Number of training examples
    m = X.shape[0]

    dj_dw = 0
    dj_db = 0

    for i in range(m):  
        f_wb = w * X[i] + b 
        dj_dw_i = (f_wb - y[i]) * X[i] 
        dj_db_i = f_wb - y[i] 
        dj_dw += dj_dw_i 
        dj_db += dj_db_i
        
    dj_dw = dj_dw / m 
    dj_db = dj_db / m 

    return dj_dw, dj_db

# compute the weights
def gradient_descent(X, y, w_in, b_in, alpha, num_iters, gradient_function):
    w = w_in
    b = b_in
    print("\nComputed weights over iterations:")
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(X, y, w , b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        if i < 10:
            num_iterations.append(i)
            # Compute the cost
            cost = compute_cost(X, y, w, b)
            cost_history.append(cost)
        
        if i % 100 == 0 or i == num_iters-1:
            print(f"Iteration {i}: ",f"w: {w: 0.3e}, b:{b: 0.5e}")
            
    # Plotting the cost against the first 10 iterations
    print("\nCost vs Iterations (Learning Curve)")
    plt.plot(num_iterations, cost_history, marker='o')
    plt.title('Cost vs. Iterations')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cost')
    plt.show()
    
    return w,b

# initialize fitting parameters.
initial_w = 0.
initial_b = 0.

# some gradient descent settings
iterations = 1500
alpha = 0.01

w_final, b_final = gradient_descent(X_train ,y_train, initial_w, initial_b, alpha, iterations, compute_gradient)
print("\nw,b found by gradient descent:", w_final, b_final)

m = X_train.shape[0]
predicted = np.zeros(m)

for i in range(m):
    predicted[i] = w_final * X_train[i] + b_final
    
# Plot the linear fit
plt.plot(X_train, predicted, c = "b")

# Create a scatter plot of the data. 
plt.scatter(X_train, y_train, marker='x', c='r') 

# Set the title
plt.title("Profits vs. Population per city")
# Set the y-axis label
plt.ylabel('Profit in $10,000')
# Set the x-axis label
plt.xlabel('Population of City in 10,000s')

# display the plot
print("\nOur model fit")
plt.show()

# test the model 
population_raw = int(input("\nEnter the city's population: "))
population = population_raw/10000

profit = compute_model_output(population, w_final, b_final)

print(f"For population = {population_raw}, we predict a profit of ${round((profit*10000),2)}")
