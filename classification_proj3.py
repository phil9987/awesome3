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
from sklearn.decomposition import PCA

from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn import preprocessing
from sklearn.neighbors.dist_metrics import DistanceMetric
from sklearn.multiclass import OneVsOneClassifier

import libfann

import h5py

pca = PCA(n_components=600)


def score(gtruth, gpred):
    # Minimizing this should result in minimizing sumscore in the end.
    # We do not actually need the len(gtruth), but it enhances debugging, since it then corresponds to the sumscore.

    return float(np.sum(gtruth != gpred))/(len(gtruth))


def sumscore(gtruth1, gtruth2, gpred1, gpred2):
    return float((np.sum(gtruth1 != gpred1) + np.sum(gtruth2 != gpred2)))/(2*len(gtruth1))

def sumscore_single(gtruth1, gpred1):
    return float(np.sum(gtruth1 != gpred1))/len(gtruth1)

def sumscore_classifier(class1, class2, X, Y):
    return sumscore(Y[:, 0], Y[:, 1], class1.prediccl.predict(X), class2.predict(X))

def sumscore_classifier_single(cl, X, Y):
    return sumscore_single(Y[:], cl.predict(X))

def sumscore_classifier_ann_single(X, Y):
    #approxFun1 =  (np.vectorize(class1))
    #approxFun2 =  (np.vectorize(class2))
    ann = libfann.neural_net()
    ann.create_from_file("project_data/ann"+str(max(Y))+".net")
    nr_features = max(X.shape)
    mat1 = np.array([])
    for i in range(nr_features):
        out_vec = ann.run(X[i,:])
        out_vec_rounded = int(np.round(out_vec))
        if len(out_vec) > 1:
            print 'error: more than one output unit was 1; there were ',len(out_vec),' 1-output units.'
            cl = np.where(out_vec==max(out_vec))[0][0]
        else:
            cl = np.where(out_vec==1)[0]
        mat1 = np.append(mat1, cl)
    return sumscore_single(Y[:, 0], mat1)


def read_path(inpath):
    X = []
    with open(inpath, 'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        for row in reader:
            X.append(map(float, row))
    return X


def read_features(X, means,stds, features_fn):
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


def some_features(x):
    return [x]
#    return [np.log(1 + np.abs(x)), np.exp(x), x, x ** 2, np.abs(x)]


# Assume that all values in x are ready-to-use features (i. e. no timestamps)
def simple_implementation(x, means,stds):
#    x = map(float, x)
    fs = [y for i in range(len(x)) for y in [(float(x[i])-means[i])/stds[i]]]
    fs.extend(x[9:])
    fs.append(1)
    return fs


def ortho(fns, x, means,stds):
    y = []
    for fn in fns:
        y.extend(fn(x, means,stds))
    return y

def convert_Y_to_discrete(Y):
    nr_datapoints = max(np.shape(Y))
    Y = np.array(Y).flatten()
    classes = np.unique(Y)
    nr_classes = len(classes)
    Y_ext = np.zeros((nr_datapoints, nr_classes))
    i = 0
    for y in Y:
        Y_ext[i, y] = 1
        i = i+1
    return Y_ext

def revert_discrete_Y(Y_ext):
    Y = np.zeros(max(np.shape(Y_ext)))
    for i in range(len(Y)):
        est = np.where(Y_ext[i]==1)[0]
        if not est.size==0:
            Y[i] = est[0]
        else:
            Y[i] == np.where(Y_ext[i]==max(Y_ext[i]))[0]
    return Y

def predict_and_print(name, class1, class2, X):
    Ypred = np.array([class1.predict(X), class2.predict(X)])
    np.savetxt('project_data/' + name + '.txt', Ypred.T, fmt='%i', delimiter=',')

def predict_and_print_single(name, cl, X):
    Ypred = np.array([cl.predict(X)]).flatten()
    np.savetxt('project_data/' + name + '.txt', Ypred.T, fmt='%i', delimiter=',')

def predict_and_print_ann(name, X):
    ann = libfann.neural_net()
    ann.create_from_file("project_data/ann9.0.net")

    Ypred = [[0,0]];
    nrFeatures = max(X.shape)
    for i in range(nrFeatures):
        out_vec = ann.run(map(float,X[i,:].tolist()[0]))
        out_vec_rounded = int(np.round(out_vec))
        if len(out_vec) > 1:
            print 'error: more than one output unit was 1; there were ',len(out_vec),' 1-output units.'
            cl = np.where(out_vec==max(out_vec))[0][0]
        else:
            cl = np.where(out_vec==1)[0][0]
        Ypred = np.append(Ypred, [cl],axis=0)

    Ypred=Ypred[1:,:]
    #X = X.tolist()
    #Ypred =[class1(X), class2(X)]
    np.savetxt('project_data/' + name + '.txt', Ypred.T, fmt='%i', delimiter=',')

def lin_classifier(Xtrain, Ytrain):
    classifier = LinearSVC()
    classifier.fit(Xtrain, Ytrain)
    print 'LIN: coef', classifier.coef_
    return classifier

def onevsone_classifier(Xtrain,Ytrain):
    classifier = OneVsOneClassifier(LinearSVC(random_state=0))
    classifier.fit(Xtrain,Ytrain)
    return classifier

def tree_classifier(Xtrain, Ytrain):
    '''param_grid = {
                  'max_features': range(20,53,15),}
        #'min_samples_split': range(2,4,1),
        #'max_depth': range(10,50,20)
    classifier = ExtraTreesClassifier(n_jobs=-1,verbose=0, min_samples_leaf=3, n_estimators = 220, criterion = 'entropy')

    classifier.fit(Xtrain, Ytrain)
    print 'TREE: classifier: ', classifier
    print 'TREE: classifier.score: ', score(Ytrain, classifier.predict(Xtrain))
    scorefun = skmet.make_scorer(lambda x, y: -score(x, y))
    grid_search = skgs.GridSearchCV(classifier, param_grid, scoring=scorefun, cv=5)
    grid_search.fit(Xtrain, Ytrain)
    print 'TREE: best_estimator_: ', grid_search.best_estimator_
    print 'TREE: best_estimator_.score: ', score(Ytrain, grid_search.predict(Xtrain))
    return grid_search.best_estimator_

    '''

    #evaluated classifier
    classifier = ExtraTreesClassifier(n_jobs=-1,
                                      criterion='entropy',
                                      max_features=100,
                                      min_samples_split=4,
                                      n_estimators=320)
    classifier.fit(Xtrain,Ytrain)
    return classifier


def forest_classifier(Xtrain, Ytrain):
    param_grid = {'n_estimators': range(1, 100, 25), 'max_features': range(1, 50, 5), 'min_samples_split': range(1,10,1), 'max_depth': range(1,1000,100)}
    classifier = RandomForestClassifier(n_jobs=-1,verbose=1, max_depth=None,min_samples_split=1)
    classifier.fit(Xtrain, Ytrain)
    print 'FOREST: classifier: ', classifier
    print 'FOREST: classifier.score: ', score(Ytrain, classifier.predict(Xtrain))
    scorefun = skmet.make_scorer(lambda x, y: -score(x, y))
    grid_search = skgs.GridSearchCV(classifier, param_grid, scoring=scorefun, cv=3)
    grid_search.fit(Xtrain, Ytrain)
    print 'FOREST: best_estimator_: ', grid_search.best_estimator_
    print 'FOREST: best_estimator_.score: ', score(Ytrain, grid_search.predict(Xtrain))
    return grid_search.best_estimator_


def knn_classifier(Xtrain, Ytrain):
    param_grid = {'n_neighbors': [4, 5, 8, 16], 'weights': ['uniform'],
#                  'metric': map(DistanceMetric.get_metric, ['manhatten', 'jaccard'])
    }
    classifier = KNeighborsClassifier(algorithm='auto')
    classifier.fit(Xtrain, Ytrain)
    print 'KNN: classifier: ', classifier
    print 'KNN: classifier.score: ', score(Ytrain, classifier.predict(Xtrain))
    scorefun = skmet.make_scorer(lambda x, y: -score(x, y))
    grid_search = skgs.GridSearchCV(classifier, param_grid, scoring=scorefun, cv=5)
    grid_search.fit(Xtrain, Ytrain)
    print 'KNN: best_estimator_: ', grid_search.best_estimator_
    print 'KNN: best_estimator_.score: ', score(Ytrain, grid_search.predict(Xtrain))
    return grid_search.best_estimator_


def svm_classifier(Xtrain, Ytrain):
    #param_grid = {'weights': ['uniform']}      #standard assumption by svm
    C_range = np.logspace(-3,3,11)
    gamma_range = np.logspace(-3,3,11)
    degree_range = range(2,4,1)
    coef0_range = np.logspace(-3,3,11)
    cv = StratifiedKFold(y=Ytrain, n_folds=3)
    param_grid = dict(gamma = gamma_range, C = C_range, degree = degree_range, coef0 = coef0_range)
    grid = GridSearchCV(svm.SVC(verbose=True, kernel = 'poly'), param_grid = param_grid, cv=cv, verbose=5, n_jobs=-1)
    #if opt == 0:
    #    classifier = svm.SVC(gamma = 0.063095734448019303, C= 15.848931924611142, degree = 3, verbose=True)
    #else:
    #    classifier = svm.SVC(gamma=0.25118864315095796,C=3.9810717055349691, degree = 3, verbose=True)
    #classifier.fit(Xtrain,Ytrain)
    grid.fit(Xtrain,Ytrain)
    print 'The best classifier is: %s' %grid.best_estimator_

    return grid.best_estimator_
    #return classifier


def principalComponents(Xtrain, Ytrain):
    X = pca.fit_transform(Xtrain, Ytrain)
    return X


def getNNData(XTrain, YTrain):
    data_size = max(XTrain.shape)
    nr_features = min(XTrain.shape)
    nr_classes = 1
    nndata=open("project_data/nndata.txt","w")
    nndata.write("%d %d %d \n" % (data_size, nr_features, nr_classes))   #header
    for i in range(data_size):
        x_sub = XTrain[i,:]
        x_arr = np.char.mod('%f', x_sub)
        x_line = " ".join(x_arr)+"\n"
        y_line = str(YTrain[i])+"\n"
        nndata.write(x_line)
        nndata.write(y_line)
    nndata.close()
    return "project_data/nndata.txt"



def getNNData_better(XTrain, YTrain): # one output neuron per class
    data_size = max(XTrain.shape)
    nr_features = min(XTrain.shape)
    classes = np.unique(YTrain)
    nr_classes = len(classes)
    nndata=open("project_data/nndata2.txt","w")
    nndata.write("%d %d %d \n" % (data_size, nr_features, nr_classes))   #header
    for i in range(data_size):
        x_sub = XTrain[i,:]
        x_arr = np.char.mod('%f', x_sub)
        x_line = " ".join(x_arr)+"\n"
        y_arr = np.zeros([1,10])
        y = int(YTrain[i])
        y_arr[y] = 1  #assume class names are numbers from 1 to nr_classes
        y_line = " ".join(y_arr)+"\n"
        nndata.write(x_line)
        nndata.write(y_line)
    nndata.close()
    return "project_data/nndata2.txt"

def neuralnet_classifier(XTrain, YTrain):
    data_size = max(XTrain.shape)
    nr_output = min(YTrain.shape)
    connection_rate = 1
    nr_hidden = 10
    desired_error = 0.01
    max_iterations = 500
    its_per_round = 100
    ann = libfann.neural_net()
    ann.create_sparse_array(connection_rate, (min(XTrain.shape), nr_hidden, nr_output)) #create...(rate, (in, hidden1, hidden2, out))
    ann.set_training_algorithm(libfann.TRAIN_QUICKPROP)
    ann.set_learning_rate(0.7)
    ann.set_activation_function_output(libfann.THRESHOLD)  #output units are 1 or 0

    datafile = getNNData(XTrain, YTrain)
    nndat_all = libfann.training_data()
    nndat_all.read_train_from_file(datafile)
    nndat_all.shuffle_train_data()
    nndat_train = libfann.training_data(nndat_all)
    nndat_test = libfann.training_data(nndat_all)
    nndat_train.subset_train_data(0, int(np.ceil(0.85*data_size)))
    nndat_test.subset_train_data(int(np.ceil(0.85*data_size)), int(np.floor(0.15*data_size)))
    #ann.set_input_scaling_params(nndat_train, -1., 1.)
    #ann.set_input_scaling_params(nndat_test, -1., 1.)
    testE = 5; # mean square error
    counter = 0; max_counter = 2;   #if more than max_counter times the error on testing data increased, stop
    overfit = False
    i = 0 #nr of runs of network updates
    while i < max_iterations & ~overfit:
        ann.train_on_data(nndat_train, its_per_round, 0, desired_error)
        trainE_new = ann.test_data(nndat_train)
        testE_new = ann.test_data(nndat_test)
        if testE_new > testE:
            counter = counter+1
            if counter > max_counter:
                overfit = True
        trainE = trainE_new
        testE = testE_new
        i = i + its_per_round
        if i % 1000 == 0:
            print 'nr of epochs so far: %d; error on training set: %f; on test set: %f' % (i, trainE, testE)
    print 'network has been trained in %d epochs; error on training set: %f; on test set: %f' % (i, trainE, testE)
    print 'overfitting: '+str(overfit)
    ann.save("project_data/ann"+str(max(YTrain))+".net")

    return lambda x: round(ann.run(x))
    #return ann



def multi_classifier(classifiers):
    return {}




# def regress(fn, name, X, Y, Xval, Xtestsub):
#     Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X, Y, train_size=0.7)
#
#     class1 = fn(Xtrain, Ytrain[:, 0])
#     class2 = fn(Xtrain, Ytrain[:, 1])
#     print 'DEBUG: classifier trained'
#
#     #print 'SCORE:', name, ' - trainset ', sumscore_classifier(class1, class2, Xtrain, Ytrain)
#     #print 'SCORE:', name, ' - test ', sumscore_classifier(class1, class2, Xtest, Ytest)
#     print 'SCORE:', name, ' - trainset ', sumscore_classifier_ann(class1, class2, Xtrain, Ytrain)
#     print 'SCORE:', name, ' - test ', sumscore_classifier_ann(class1, class2, Xtest, Ytest)
#
#     #predict_and_print('validate_y_' + name, class1, class2, Xval)
#     #predict_and_print('test_y_' + name, class1, class2, Xtestsub)
#     predict_and_print_ann('validate_y_' + name, class1, class2, Xval)
#     predict_and_print_ann('test_y_' + name, class1, class2, Xtestsub)




def regress_hdf5(fn, name, X, Y, Xval, Xtestsub):
    Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X, Y, train_size=0.8)

    cl = fn(Xtrain, Ytrain)
    print 'DEBUG: classifier trained'


    #Ytrain = revert_discrete_Y(Ytrain)
    Ypred = cl.predict(Xtrain)
    Ypred_test = cl.predict(Xtest)
    #Ypred = revert_discrete_Y(Ypred)
    #Ypred_test = revert_discrete_Y(Ypred_test)

    #print 'SCORE:', name, ' - trainset ', sumscore_classifier(class1, class2, Xtrain, Ytrain)
    #print 'SCORE:', name, ' - test ', sumscore_classifier(class1, class2, Xtest, Ytest)
    #print 'SCORE:', name, ' - trainset ', sumscore_classifier_single(cl, Xtrain, Ytrain)
    #print 'SCORE:', name, ' - test ', sumscore_classifier_single(cl, Xtest, Ytest)
    print 'SCORE:', name, ' - trainset ', sumscore_single(Ytrain.flatten(), Ypred)
    print 'SCORE:', name, ' - test ', sumscore_single(Ytest.flatten(), Ypred_test)
    #print 'SCORE:', name, ' - trainset ', sumscore_classifier_ann_single(Xtrain, Ytrain)
    #print 'SCORE:', name, ' - test ', sumscore_classifier_ann_single(Xtest, Ytest)

    predict_and_print_single('validate_y_' + name, cl, Xval)
    predict_and_print_single('test_y_' + name, cl, Xtestsub)
    #predict_and_print_ann('validate_y_' + name, Xval)
    #predict_and_print_ann('test_y_' + name, Xtestsub)



def regress_no_split(fn, name, X, Y, Xval, Xtestsub):
    class1 = fn(X, Y[:, 0])
    class2 = fn(X, Y[:, 1])

    print 'SCORE:', name, ' - all ', sumscore_classifier(class1, class2, X, Y)

    ''' score_fn = skmet.make_scorer(score)
    scores = skcv.cross_val_score(class1, X, Y[:, 0], scoring=score_fn, cv=3)
    print 'SCORE:', name, ' - (cv) mean on 1 : ', np.mean(scores), ' +/- ', np.std(scores)
    scores = skcv.cross_val_score(class2, X, Y[:, 1], scoring=score_fn, cv=3)
    print 'SCORE:', name, ' - (cv) mean on 2 : ', np.mean(scores), ' +/- ', np.std(scores)
    '''
    predict_and_print('validate_y_' + name, class1, class2, Xval)
    predict_and_print('test_y_' + name, class1, class2, Xtestsub)


def read_and_regress_hdf5(feature_fn):
    train_file = h5py.File("./project_data/train.h5", "r")
    for name in train_file:
        print name
    train_data = train_file['data'][:]
    train_label = train_file['label'][:]
    print 'shape of train-data: ',train_data.shape
    print 'shape of train-label: ',train_label.shape
    Xo = np.array(train_data)
    Y = np.array(train_label)
    Y_big = convert_Y_to_discrete(Y)
    Xo = Xo[0:5000,:]
    Y_big = Y_big[0:5000]
    Y = Y[0:5000]
    X = principalComponents(Xo, Y)

    #means = [np.mean(Xo[:,i]) for i in range(np.shape(Xo)[1])]
    #print 'means: ', means
    #stds = [np.std(Xo[:,i]) for i in range(np.shape(Xo)[1])]
    #print 'stds: ', stds

    #X = read_features(Xo, means, stds, feature_fn)

    train_file = h5py.File("./project_data/validate.h5", "r")
    Xvalo = train_file['data'][:]
    Xval = pca.transform(Xvalo)
    #Xval = read_features(Xvalo, means, stds, feature_fn)
    train_file = h5py.File("./project_data/test.h5", "r")
    Xtesto = train_file['data'][:]
    Xtest = pca.transform(Xtesto)
    #Xtest = read_features(Xtesto, means, stds, feature_fn)

    #regress_hdf5(neuralnet_classifier, 'ann', X, Y, Xval, Xtest)
    regress_hdf5(tree_classifier, 'extra_trees', X, Y, Xval, Xtest)


if __name__ == "__main__":
    read_and_regress_hdf5(lambda x, means, stds: ortho([simple_implementation], x, means, stds))
