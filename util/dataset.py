import scipy.io
import numpy as np
import tensorflow as tf

class Dataset:
    def __init__(self):
        self.path1 = "Network class/cls1.mat"
        self.path2 = "Network class/cls2.mat"
        self.path3 = "Network class/cls3.mat"
        self.load_data()
        self.y = np.zeros((self.data.shape[0], 3))
        self.y[0:self.data1.shape[0], 0] = 1
        self.y[self.data1.shape[0]:self.data1.shape[0]+self.data2.shape[0], 1] = 1
        self.y[self.data1.shape[0]+self.data2.shape[0]:, 2] = 1
        self.data = tf.convert_to_tensor(self.data, dtype=tf.float32)
        self.y = tf.convert_to_tensor(self.y, dtype=tf.float32)
        
        
    def load_data(self):
        # Implement code to load data from the file
        # Your code here
        self.data1 = scipy.io.loadmat(self.path1)["cls1"]
        self.data2 = scipy.io.loadmat(self.path2)["cls2"]
        self.data3 = scipy.io.loadmat(self.path3)["cls3"]
        self.data = np.concatenate((self.data1, self.data2, self.data3), axis=0)
        
        
    def __len__(self):
        # Implement code to return the length of the dataset
        # Your code here
        return self.data.shape[0]
        
    def __getitem__(self, index):
        # Implement code to get an item from the dataset given an index
        # Your code here
        return self.data[index,:], self.y[index,:]
    
    def to_tfdataset(self):
        # Convert the numpy arrays to TensorFlow datasets
        tf_data = tf.data.Dataset.from_tensor_slices((self.data, self.y))
        return tf_data


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    mydataset = Dataset()
    plt.plot(mydataset[0][0])
    plt.show()
    print(mydataset[0][1])
    print(len(mydataset))
    