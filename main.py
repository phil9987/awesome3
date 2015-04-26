import h5py
import numpy as np

def main():
    validate = h5py.File("./julia_src/project_data/validate_double.h5")
    train_file = h5py.File("./julia_src/validate_output.h5", "r")
    for name in validate:
        print name
    for name in train_file:
        print name

    #train_data = train_file['data'][:]
    train_label = train_file['label'][:]
    #print 'shape of train-data: ',train_data.shape
    print 'shape of train-label: ',train_label.shape
    validate_data = validate['data'][:]
    print 'shape of validate: ', validate_data.shape
    print train_label
    #print train_data
    #print train_label


if __name__ == "__main__":
    main()