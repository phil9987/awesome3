import h5py
import numpy as np

def main():
    train_file = h5py.File("./project_data/train.h5", "r")
    for name in train_file:
        print name
    train_data = train_file['data'][:]
    train_label = train_file['label'][:]
    print 'shape of train-data: ',train_data.shape
    print 'shape of train-label: ',train_label.shape


if __name__ == "__main__":
    main()