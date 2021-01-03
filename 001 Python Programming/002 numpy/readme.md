# Basic Python Programming

## Using Numpy library

Using numpy library to treat numerical data in matrix form will be a daily job when you become a data scientist. In this course, we are going to read 2D MNIST dataset (from the previous course) as numpy arrays. Subsequently, we will split the data into train, validation, and test dataset and save those files as .npz files.

To do this, several libraries are required:

1. os, glob : to read and write the files
2. csv : to read and write csv files
3. numpy : to use numpy arrays
4. random : to shuffle the given array

## Process

1. Read the converted 2d dataset files in 'outputs' directory in the previous course.
2. Create empty 3D numpy arrays to save loaded 2D MNIST data.
   1. The empty numpy array could be filled with Nones or zeroes. 
   2. The shape of the initialized numpy array will be (number of files, width, height).
   3. Create additional 1D numpy arrays to save the label of the loaded 2D MNIST data.
3. Using for loops, replace the empty elements in the numpy arrays with the loaded 2D MNIST data and its label.
   1. Using append or any other method to expand the size of the list or array may require additional computation time. 
   2. Therefore, we initialize the array with zeroes and replace the elements afterwards.
4. Concat the train and test arrays into one array, so that we can split into three different arrays. (train, valid, test)
5. Using the random library, shuffle the numpy array.
   1. Note that the numpy arrays for both data and label should be shuffled in the same order.
6. Split the numpy array into train, validation, and test arrays. Each array has a raio of 7:2:1 resepectively.
7. Save the splitted arrays into .npz files.
   1. The saved data should be 'train_x', 'train_y', 'valid_x', 'valid_y', 'test_x', and 'test_y'.

## Important codes

### Create an empty numpy array.

```python
import os, glob
import numpy as np

path_to_train_2d_datasets = os.path.join('..','001 csv','outputs','train','*.csv')
train_2d_files = glob.glob(path_to_train_2d_datasets)

# We already know that the width and height of the 2D MNIST dataset is 28.
train = np.empty((len(train_2d_files),28,28))

for data_idx, data_path in enumerate(train_2d_files):
    # Read the csv and replace the elements
    csv_data = # may need csv library to load the data
    for i in range(28):
        for j in range(28):
            train[data_idx,i,j] = csv_data[i][j]

```

### Concat, shuffle and split the numpy array

```python
import numpy as np

a = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
b = np.array([[5, 6]])

c = np.concatenate((a, b), axis=0)

np.random.shuffle(c) # Note that we don't get the returned value.

[d, e] = np.split(c, [3,1,1]) # Split ratio. Make sure the values should be the integers (indices). 
```

### Save and load files with npz

```python
import os
import numpy as np

x = np.array([0, 1, 2])
y = np.array([3, 4, 5])

path_to_save = os.path.join("dataset.npz")
np.savez(path_to_save, data=x, label=y)

del x
del y

data = np.load(path_to_save)

x = data["x"]
y = data["y"]
```

## Author
Hwanmoo Yong