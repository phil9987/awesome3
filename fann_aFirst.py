import numpy as np
import csv
import sklearn.linear_model as sklin
import sklearn.ensemble as rf
import sklearn.metrics as skmet
import sklearn.cross_validation as skcv
import sklearn.grid_search as skgs
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn import preprocessing
from sklearn.neighbors.dist_metrics import DistanceMetric
from sklearn.multiclass import OneVsOneClassifier

import libfann

import h5py

# parameters
normalize = True
split_training_data = True
split_percentage_train = 0.95
split_percentage_test = 0.05



def ortho(fns, x, means,stds):
    y = []
    for fn in fns:
        y.extend(fn(x, means,stds))
    return y

def simple_implementation(x, means,stds):
#    x = map(float, x)
    fs = [y for i in range(len(x)) for y in [((float(x[i])-means[i])/stds[i])]]
    #fs.extend(x[9:])
    #fs.append(1)
    return fs

def sumscore_single(gtruth1, gpred1):
    return float(np.sum(gtruth1 != gpred1))/len(gpred1)


def sumscore_classifier_ann_single(X, Y):
    #approxFun1 =  (np.vectorize(class1))
    #approxFun2 =  (np.vectorize(class2))
    ann = libfann.neural_net()
    ann.create_from_file("project_data/ann"+str(max(Y))+".net")
    nr_features = max(X.shape)
    mat1 = np.array([])
    for i in range(nr_features):
        out_vec = ann.run(X[i,:])
        out_vec_rounded = map(int, np.round(out_vec))
        out_vec_rounded = np.array(out_vec_rounded)
        cl = np.where(out_vec_rounded==1)[0]
        if not len(cl)==1:
            #print 'error in sumscore: more than one or no output unit was 1; there were ',len(cl),' 1-output units.'
            cl = np.where(out_vec==np.max(out_vec))[0][0]
        mat1 = np.append(mat1, cl)
    ann.destroy()
    return sumscore_single(Y[:, 0], mat1)


def read_features(X, means, stds, features_fn):
    M = []
    x_rows = len(X)
    i = 1
    for x in X:
        m = features_fn(x, means, stds)
        M.append(map(float,m))
        if i % 100000 == 0:
            print str(i) + ' of ' + str(x_rows) + ' rows processed...'
        i += 1
    print str(i) + ' of ' + str(x_rows) + ' rows processed...'
    return np.matrix(M)


def extract_features(feature_fn):
    train_file = h5py.File("./project_data/train.h5", "r")
    for name in train_file:
        print name
    train_data = np.array(train_file['data'][:])
    train_label = np.array(train_file['label'][:])
    print 'shape of train-data: ',train_data.shape
    print 'shape of train-label: ',train_label.shape
    Xo = train_data
    Y = train_label
    if normalize:
        means = [np.mean(Xo[:,i]) for i in range(np.shape(Xo)[1])]
        print 'means: ', means
        stds = [np.std(Xo[:,i]) for i in range(np.shape(Xo)[1])]
        print 'stds: ', stds
        X = read_features(train_data, means, stds, feature_fn)
    else:
        X = Xo
    train_file_val = h5py.File("./project_data/validate.h5", "r")
    train_file_test = h5py.File("./project_data/test.h5", "r")
    Xval = np.matrix(train_file_val['data'][:])
    Xtest = np.matrix(train_file_test['data'][:])
    if normalize:
        Xtest = read_features(Xtest, means, stds, feature_fn)
        Xval = read_features(Xval, means, stds, feature_fn)
    return X, Y, Xtest, Xval


def predict_and_print_ann(name, X, nnfilename):
    ann = libfann.neural_net()
    ann.create_from_file(nnfilename)

    Ypred = [0];
    nrFeatures = max(X.shape)
    for i in range(nrFeatures):
        out_vec = ann.run(X[i,:])
        out_vec_rounded = map(int, np.round(out_vec))
        out_vec_rounded = np.array(out_vec_rounded)
        cl = np.where(out_vec_rounded==1)[0]
        if not len(cl)==1:
            #print 'error in predict&print: more than one or no output unit was 1; there were ',len(cl),' 1-output units.'
            cl = np.where(out_vec==np.max(out_vec))[0][0]
        Ypred = np.append(Ypred, [cl], axis=0)

    Ypred=Ypred[1:]
    #X = X.tolist()
    #Ypred =[class1(X), class2(X)]
    np.savetxt('project_data/' + name + '.txt', Ypred.T, fmt='%i', delimiter=',')
    ann.destroy()


def getNNData_better(XTrain, YTrain): # one output neuron per class
    data_size = max(XTrain.shape)
    nr_features = min(XTrain.shape)
    classes = np.unique(YTrain)
    nr_classes = len(classes)
    datafile = "project_data/nndata_normalized.txt"
    nndata=open(datafile,"w")
    nndata.write("%d %d %d \n" % (data_size, nr_features, nr_classes))   #header
    for i in range(data_size):
        x_sub = XTrain[i,:]
        x_str_arr = np.char.mod('%f', x_sub)
        x_line = " ".join(x_str_arr)+"\n"
        y_arr = np.zeros([10])
        y = int(YTrain[i])
        y_arr[y] = 1  #assume class names are numbers from 0 to nr_classes-1
        y_str_arr = np.char.mod('%f', y_arr)
        y_line = " ".join(y_str_arr)+"\n"
        nndata.write(x_line)
        nndata.write(y_line)
        if i % 10000 == 0:
            print str(i)+' of '+str(data_size)+' data points written to file'
    print str(i+1)+' of '+str(data_size)+' data points written to file'
    nndata.close()
    return datafile



def create_and_train_nn(XTrain, YTrain, datafile, nr_its):
    data_size = max(XTrain.shape)
    nr_output = len(np.unique(YTrain))
    connection_rate = 1
    steepness_out = 1.2
    #nr_hidden1 = 75
    #nr_hidden2 = 51
    nr_hidden = 100
    desired_error = 0.01
    max_iterations = 3*nr_its
    its_per_round = nr_its
    ann = libfann.neural_net()
    ann.create_sparse_array(connection_rate, (min(XTrain.shape), nr_hidden, nr_output)) #create...(rate, (in, hidden1, hidden2, out))
    ann.set_training_algorithm(libfann.TRAIN_QUICKPROP)
    ann.set_activation_function_output(libfann.SIGMOID)  #output units are between 1 or 0; for execution, use libfann.THRESHOLD
    ann.set_activation_steepness_hidden(steepness_out)


    nndat_all = libfann.training_data()
    nndat_all.read_train_from_file(datafile)
    if split_training_data:
        nndat_all.shuffle_train_data()
        nndat_train = libfann.training_data(nndat_all)
        nndat_test = libfann.training_data(nndat_all)
        nndat_train.subset_train_data(0, int(np.ceil(split_percentage_train*data_size)))
        nndat_test.subset_train_data(int(np.ceil(split_percentage_train*data_size)), min(int(np.floor(split_percentage_test*data_size)), int(np.floor((1-split_percentage_train)*data_size))))
    else:
        nndat_train = nndat_all
    #ann.set_input_scaling_params(nndat_train, -1., 1.)
    #ann.set_input_scaling_params(nndat_test, -1., 1.)
    testE = 5; # mean square error
    counter = 0; max_counter = 2;   #if more than max_counter times the error on testing data increased, stop
    overfit = False
    i = 0 #nr of runs of network updates

    ## statistics on ann.train_on_data:
    #
    # ~2000 input units, 10 output units, output function: sigmoid
    #   - 1 hidden layer of 20 units, 100 epochs:
    #       TRAIN_QUICKPROP:        1 min   , error: ~0.06
    #       TRAIN_RPROP:            < 5 sec , error: ~0.09
    #       TRAIN_RPROP, 2nd run:   1 min , error:   ~0.08
    #   - 1 hidden layer of 20 units, 1000 epochs:
    #       TRAIN_RPROP:        stopped it after 3 or 4 mins



    while i < max_iterations & ~overfit:
        ann.train_on_data(nndat_train, its_per_round, 0, desired_error)
        trainE_new = ann.test_data(nndat_train)
        if split_training_data:
            testE_new = ann.test_data(nndat_test)
            if testE_new > testE:
                counter = counter+1
                if counter > max_counter:
                    overfit = True
            trainE = trainE_new
            testE = testE_new
        i = i + its_per_round
        #if i % 10 == 0:
        print 'nr of epochs so far: %d; error on training set: %f; on test set: %f' % (i, trainE, testE)
    print 'network has been trained in %d epochs; error on training set: %f; on test set: %f' % (i, trainE, testE)
    print 'overfitting: '+str(overfit)+" (test data exists: "+str(split_training_data)+")"
    ann.save("project_data/ann"+str(max(YTrain))+".net")

    nndat_all.destroy_train()
    nndat_train.destroy_train()
    nndat_test.destroy_train()

    return "ann"+str(max(YTrain))
    #return lambda x: round(ann.run(x))
    #return ann









def read_and_regress_hdf5(feature_fn):
    X, Y, Xtest, Xval = extract_features(feature_fn)
    Xtrain, Xtrain_test, Ytrain, Ytrain_test = skcv.train_test_split(X, Y, train_size=0.8)

    datafile = getNNData_better(X, Y)    #You only need to do this once. Just needed 54 seconds, so don't if you don't need to
    #datafile = "project_data/nndata.txt"

    name = create_and_train_nn(Xtrain, Ytrain, datafile, 10)

    #test it
    print 'SCORE:', name, ' - trainset ', sumscore_classifier_ann_single(Xtrain, Ytrain)
    print 'SCORE:', name, ' - test ', sumscore_classifier_ann_single(Xtrain_test, Ytrain_test)

    predict_and_print_ann('validate_y_' + name, Xval, name)
    predict_and_print_ann('test_y_' + name, Xtest, name)


    #name = create_and_train_nn(Xtrain, Ytrain, datafile, 150)

    #test it
    #print 'SCORE:', name, ' - trainset ', sumscore_classifier_ann_single(Xtrain, Ytrain)
    #print 'SCORE:', name, ' - test ', sumscore_classifier_ann_single(Xtrain_test, Ytrain_test)

    #name = create_and_train_nn(Xtrain, Ytrain, datafile, 150)

    #test it
    #print 'SCORE:', name, ' - trainset ', sumscore_classifier_ann_single(Xtrain, Ytrain)
    #print 'SCORE:', name, ' - test ', sumscore_classifier_ann_single(Xtrain_test, Ytrain_test)


    #predict_and_print_ann('validate_y_' + name, Xval, name)
    #predict_and_print_ann('test_y_' + name, Xtest, name)








if __name__ == "__main__":
    read_and_regress_hdf5(lambda x, means, stds: ortho([simple_implementation], x, means, stds))
