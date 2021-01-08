# Basic Python Programming

## Custom Dataset

Many tutorials on the internet provide pre-built datasets for the training such as MNIST, CIFAR-10, COCO, etc. However, for our own research, we may need to create our own dataset. In this course, we are going to create custom datasets for the classification and regression.

There is no right way of creating a custom dataset, but the important thing is that we should know how to manipulate the collected data to right purpose. When data is provided appropriately according to its characteristics, the performance of neural networks may improve. (e.g. sequential data to the LSTM networks, spatial information to the CNN networks, etc.)

## Process

1. Choose a topic. The topic could be like 'cats vs. dogs' (classification) or regression model which estimates/predicts data from input data. 
2. Collect the data. Not much, only enough to use as an example (maybe 10).
3. Try to write a code using 'Class'. 
   1. It will be much more convinient when dealing with larger projects.
   2. Please refer to the example code ('data_loader_example.py').
      1. The example code imports multiple csv files which are the sensory outputs from the acceleration and velocity sensors attached at the suspensions of the full car.
      2. The output values (a, K, G..) are obtained from the matlab simulation.
      3. The sensor outputs are windowed so that we can feed sequential data (sensory outputs) into the neural network. (e.g. at t_n, we input the measurements from t_{n-w+1} to t_n.)

## Author
Hwanmoo Yong