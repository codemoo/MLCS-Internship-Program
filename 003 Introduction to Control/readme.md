# Basic Python Programming

## Handling csv files

Reading and writing csv files is one of the basic functions that we use a lot when dealing with data. In this course, we are going to read the MNIST dataset in the form of 1D vector and convert it into 2D vector form. 

To do this, several libraries are required:

1. os : to read and write the files
2. csv : to read and write csv files.

## Process

1. Read the dataset files in 'datasets' directory using relative path.
2. Read the lines (rows) in the csv and split the string into a label and image data.
3. Convert the image data which is in 1D vector form into 2D vector using 'for' loops.
4. Create the 'outputs/train' and 'outputs/test' directories, if not exists.
5. Save each individual 2D vector image into '#{label}-#{row_index}.csv' under train or test outputs directory.

## Important codes

### Relative path to the dataset files

```python
import os

path_to_train_dataset_1 = os.path.join('datasets','mnist_train.csv')
path_to_train_dataset_2 = os.path.join('datasets','mnist_train_2.csv')
path_to_test_dataset = os.path.join('datasets','mnist_test.csv')

paths_to_datasets = os.path.join('datasets','*.csv') # Returns list of paths
```

### Reading and writing csv file using csv library

```python
import os
import csv

path_to_train_dataset_1 = os.path.join('datasets','mnist_train.csv')
path_to_output = os.path.join('datasets','mnist_train_cloned.csv')

output_file = open(path_to_output, 'w')
csv_writer = csv.writer(output_file)
csv_writer.writerow(['Label', 'Pixel 1',' Pixel 2'])
with open(path_to_train_dataset_1, 'r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        print(row)
        csv_writer.writerow(row)

output_file.close
        
```

### Create directory if not exists

```python
import os

output_path_train = os.path.join('outputs', 'train')

if not os.path.exists(output_path_train):
    os.makedirs(output_path_train)
```

## Author
Hwanmoo Yong