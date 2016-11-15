Assignment 04 (Due date: Oct. 16, 2016)
Neural Networks
Assignment 04
Due date: Oct. 16, 2016

The purpose of the this assignment is to practice with Widrow-Huff learning and adaptive filters.

Write a program to predict the price and the volume of a stock
Apply the LMS algorithm
Read the historical data which provides price and volume
Create a slider to select the number of delayed elements for price and volume.
Crate a slider to adjust the learning rate.
Create a "Set Weights to Zero" button. When this button is pushed all the weights should be set to zero.
Create a "Sample Size Percentage" slider to allow the user to selectthe number of samples to be used. range 0 to 100. Default value should be 100 which means to select the entire data set.
Create a "Batch Size" slider to allow the user to change the number of training samples before the error is calculated and displayed.
Create a "Number of Iterations" slider to allow the user to change the number of times that the system goes over all the samples. Range 1 to 10
Create "Adjust Weights (Learn)" button such that each time this button is pressed the  learning rule is applied ("Batch Size" samples should be processed).
Plot the Mean Squared Error (MSE) and maximum absolute error for price and volume after each "Batch Size" on the same graph. Calculations of the error should be done after the entire "Batch Size" samples have been processed. In other words, go through each sample in the current batch, adjust the weight accordingly. Once the entire batch is processed, go back to the start of that batch, calculate the error for each sample and then calculate the MSE and max absolute error for that batch.

Notes:
Your neural network should have two nodes (Price, Volume).
The weights should not be reset when the "Learn" button is pressed.


Clarification:
If the "Number of Delayed elements" is equal to 7 then your network will have 16 input values and two outputs (one price and 7 previous prices plus one volume and 7 previous volumes).
if the "Sample Size"=1000 , "Batch Size"=200, and "Number of Iterations"=6 ; Then you should select the first 1000 samples and ignore the rest). You should process one sample at a time. After processing every 200 samples you should calculate the mean and max error for those 200. samples. You should go over the 1000 samples 6 times. In other words you should process 6000 samples and calculate and display the error after every 200 samples.
