import h5py
import numpy as np

def main():
    train_file = h5py.File("./julia_src/project_data/train_label.h5", "r")
    conv_file = h5py.File("./julia_src/project_data/train_conv.h5", "w")
    for name in train_file:
        print name
    #train_data = train_file['data'][:]
    train_label = train_file['label'][:]
    #print 'shape of train-data: ',train_data.shape
    print 'shape of train-label: ',train_label.shape
    print train_label
    #print train_data
    #print train_label


if __name__ == "__main__":
    main()