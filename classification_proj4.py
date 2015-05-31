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
from sklearn import mixture

import time

#import libfann

#import h5py

pca = PCA(n_components=200)
big_Y = False

def score(gtruth, gpred):
    # Minimizing this should result in minimizing sumscore in the end.
    # We do not actually need the len(gtruth), but it enhances debugging, since it then corresponds to the sumscore.

    return float(np.sum(gtruth != gpred))/(len(gtruth))

def logscore(gtruth, gpred):
    #probabilities_that_count = np.sum((gtruth*gpred), axis=1)
    nr_data = (gtruth.shape)[0]
    compare = np.zeros(nr_data) + 0.0001
    return float(np.sum(-np.log(np.maximum(compare, np.sum((gtruth*gpred), axis=1))))) / nr_data

def sumscore(gtruth1, gtruth2, gpred1, gpred2):
    return float((np.sum(gtruth1 != gpred1) + np.sum(gtruth2 != gpred2)))/(2*len(gtruth1))

def sumscore_classifier(class1, class2, X, Y):
    return sumscore(Y[:, 0], Y[:, 1], class1.prediccl.predict(X), class2.predict(X))

def sumscore_classifier_single(cl, X, Y):
    return score(Y[:], cl.predict(X))


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


def normalize(X, means, stds):
    Xn = np.zeros_like(X)
    #Xn = np.zeros_like(X[:,0])
    for i in range(min(X.shape)):
        #np.concatenate(Xn,(map(float,(X[:,i])-means[i]))/stds[i])
        Xn[:,i] =  np.array([(map(float,(X[:,i])-means[i]))/stds[i]]).transpose()
        #Xn = np.column_stack((Xn,(map(float,(X[:,i])-means[i]))/stds[i]))
    Xn = Xn[:,1:]
    return Xn

def normalize_rows(Y):
    row_sums = np.sum(Y,axis=1)
    return Y/row_sums[:,np.newaxis]*0.9999

def invert_perm(perm):
    p_inv = np.zeros_like(perm)
    j = 0
    for i in perm:
        p_inv[i] = j
        j = j+1
    return p_inv




def reduce_result_to_real_classes(Ypredicted, perm, nr_classes):
    #uses original permutation, the one returned together with the classifier
    #perm[i] sais which real class fits best to predicted class i
    Y_pred_red = np.zeros((Ypredicted.shape[0], nr_classes))
    for i in range(nr_classes):
        Y_pred_red[:,i] = np.sum(Ypredicted[:,np.where(perm==i)][:,0,:], axis=1)
        Y_pred_red[:,i] += 1./float(nr_classes)*np.sum(Ypredicted[:,np.where(perm==-1)][:,0,:], axis=1)
    return Y_pred_red


def ortho(fns, x, means,stds):
    y = []
    for fn in fns:
        y.extend(fn(x, means,stds))
    return y

def convert_Y_to_discrete(Y):
    nr_datapoints = max(np.shape(Y))
    Y = np.array(Y).flatten()
    classes = np.unique(Y)
    nr_classes = 8#len(classes)
    Y_ext = np.zeros((nr_datapoints, nr_classes))
    i = 0
    for y in Y:
        Y_ext[i, y] = 1.
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

def predict_and_print_probabs(name, cl, X, perm, nr_classes):
    Ypred = cl.predict_proba(X)
    Ypred = reduce_result_to_real_classes(Ypred, perm, nr_classes)
    np.savetxt('project_data/' + name + '.txt', Ypred.T, fmt='%f', delimiter=',')

def predict_and_print_single(name, cl, X):
    if big_Y:
        Ypred = cl.predict(X)
        Ypred = np.array([revert_discrete_Y(Ypred)]).flatten()
    else:
        Ypred = np.array([cl.predict(X)]).flatten()
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


def a_GMM_classifier(Xtrain, Ytrain, gmm_type, each_class_once=False):
    # gmm_type: e.g. mixture.GMM
    nr_classes = len(np.unique(Ytrain))-1 #all except -1
    if each_class_once:
        n = nr_classes
    else: n = nr_classes*4
    classifier = gmm_type(n_components=n)
    classifier.means_ = np.array([Xtrain[Ytrain==i].mean(axis=0) for i in range(nr_classes+1)])
    classifier.fit(Xtrain[0:-1,:])
    if each_class_once:
        classes_predicted = range(nr_classes)
    else: classes_predicted = range(np.max(np.unique(classifier.predict(Xtrain[0:-1,:])))+1)
        # find permutation of classes returned by classifier to fit best to its datas' labels
    Ylab_real = Ytrain[Ytrain != -1]
    Ylab_pred = classifier.predict(Xtrain[Ytrain != -1])
    perm = []
    best_classes = []
    ratio_best_class = []
    other_best_class = list(np.zeros(nr_classes)-1)
    for cl in classes_predicted:
        classlabels = Ylab_real[Ylab_pred == cl]
        if not len(classlabels)==0:
            occurences = np.array([len(np.where(classlabels==a_class)[0]) for a_class in range(nr_classes)])
            best_classes = np.where(occurences == max(occurences))[0]
            ratio = float(occurences[best_classes[0]])/len(classlabels)
            ratio_best_class.append(ratio)
            if ratio <= 0.5:
                perm.append(-1)
            else: perm.append(best_classes[0])
        else:
            #if each_class_once:
            perm.append(-1)
            #else: perm.append(1)
            ratio_best_class.append(0.0)
        if not len(best_classes)<=1:
            other_best_class.append(best_classes[1])
    perm = np.array(perm)
    if each_class_once:
        for cl in range(nr_classes):
            predictor_classes = np.where(perm==cl)[0]
            if len(predictor_classes)>1:
                nr_too_much = len(predictor_classes)-1
                for i in range(nr_too_much):
                    for other_cl in range(nr_classes):
                        if not perm.__contains__(other_cl):
                            if ratio_best_class[predictor_classes[0]] >= ratio_best_class[predictor_classes[i+1]]:
                                perm[predictor_classes[i+1]] = other_cl
                            else:   #move best class to the front of predictor_classes, so that the best one is in predictor_classes[0] in the end
                                helper = predictor_classes[0]
                                predictor_classes[0] = predictor_classes[i+1]
                                predictor_classes[i+1] = helper
                                perm[predictor_classes[i+1]] = other_cl
                            break
            #perm: perm[i] = good real class corresponding to predicted class i
    return classifier, perm

# def GMM_classifier_old(Xtrain, Ytrain):
#     nr_classes = len(np.unique(Ytrain))-1 #all except -1
#     classifier = mixture.GMM(n_components=nr_classes)
#     classifier.means_ = np.array([Xtrain[Ytrain==i].mean(axis=0) for i in range(nr_classes+1)])
#     classifier.fit(Xtrain[0:-1,:])
#         # find permutation of classes returned by classifier to fit best to its datas' labels
#     Ylab_real = Ytrain[Ytrain != -1]
#     Ylab_pred = classifier.predict(Xtrain[Ytrain != -1])
#     perm = []
#     ratio_best_class = []
#     other_best_class = list(np.zeros(nr_classes)-1)
#     for cl in range(nr_classes):
#         classlabels = Ylab_real[Ylab_pred == cl]
#         if not len(classlabels)==0:
#             occurences = np.array([len(np.where(classlabels==a_class)[0]) for a_class in range(nr_classes)])
#             best_classes = np.where(occurences == max(occurences))[0]
#             perm.append(best_classes[0])
#             ratio_best_class.append(float(occurences[best_classes[0]])/len(classlabels))
#         else:
#             perm.append(-1)
#             ratio_best_class.append(0.0)
#         if len(best_classes)!=1:
#             other_best_class.append(best_classes[1])
#     perm = np.array(perm)
#     for cl in range(nr_classes):
#         predictor_classes = np.where(perm==cl)[0]
#         if len(predictor_classes)>1:
#             nr_too_much = len(predictor_classes)-1
#             for i in range(nr_too_much):
#                 for other_cl in range(nr_classes):
#                     if not perm.__contains__(other_cl):
#                         if ratio_best_class[predictor_classes[0]] >= ratio_best_class[predictor_classes[i+1]]:
#                             perm[predictor_classes[i+1]] = other_cl
#                         else:   #move best class to the front of predictor_classes, so that the best one is in predictor_classes[0] in the end
#                             helper = predictor_classes[0]
#                             predictor_classes[0] = predictor_classes[i+1]
#                             predictor_classes[i+1] = helper
#                             perm[predictor_classes[i+1]] = other_cl
#                         break
#         #perm: perm[i] = good real class corresponding to predicted class i
#     return classifier, perm

def DPGMM_classifier(Xtrain, Ytrain):
    classifier, perm = a_GMM_classifier(Xtrain, Ytrain, mixture.DPGMM, False)
    return classifier, perm

def VBGMM_classifier(Xtrain, Ytrain):
    classifier, perm = a_GMM_classifier(Xtrain, Ytrain, mixture.VBGMM, True)
    return classifier, perm

def GMM_classifier(Xtrain, Ytrain):
    classifier, perm = a_GMM_classifier(Xtrain, Ytrain, mixture.GMM, True)
    return classifier, perm

def principalComponents(Xtrain, Ytrain):
    X = pca.fit_transform(Xtrain, Ytrain)
    return X



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




def regress(fn, name, X, Y, Xval, Xtestsub):

    #Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X, Y, train_size=0.99)

    test_perc = 0.1
    labeled_indices = np.where(Y!=-1)[0]
    test_idx = labeled_indices[:int(np.floor(test_perc*len(labeled_indices)-1.))]
    #train_idx = labeled_indices[np.floor(test_perc*len(labeled_indices)):]

    X_labeled = X[labeled_indices]
    Y_labeled = Y[labeled_indices]
    nr_classes = len(np.unique(Y_labeled))

    Xtrain = (np.delete(X,test_idx,axis=0))
    Xtest = X[test_idx]
    Ytrain = np.delete(Y,test_idx,axis=0)
    Ytest = Y[test_idx]

    t0 = time.clock()
    cl, perm = fn(Xtrain,Ytrain)
    t1 = time.clock()
    print 'DEBUG: classifier trained; time: ',str(t1-t0), ' s'

    #perm = invert_perm(perm)
    Ypred      = cl.predict_proba(X_labeled)
    Ypred_test = cl.predict_proba(Xtest)
    Ypred = reduce_result_to_real_classes(Ypred, perm, nr_classes)
    Ypred_test = reduce_result_to_real_classes(Ypred_test, perm, nr_classes)
    #Ypred      = Ypred[:,perm][:,:len(np.unique(Y_labeled))]
    #Ypred_test = Ypred_test[:,perm][:,:len(np.unique(Y_labeled))]
    #Ypred      = normalize_rows(Ypred)
    #Ypred_test = normalize_rows(Ypred_test)
    Y_labeled  = convert_Y_to_discrete(Y_labeled.flat)
    Ytest      = convert_Y_to_discrete(Ytest.flat)

    print 'SCORE:', name, ' - trainset ', logscore(Y_labeled, Ypred)
    #print 'SCORE:', name, ' - trainset ', sumscore_classifier_ann_single(Xtrain, Ytrain)
    print 'SCORE:', name, ' - test ', logscore(Ytest, Ypred_test)

    predict_and_print_probabs('validate_y_' + name, cl, Xval, perm, nr_classes)
    predict_and_print_probabs('test_y_' + name, cl, Xtestsub, perm, nr_classes)
    #predict_and_print_ann('validate_y_' + name, Xval)
    #predict_and_print_ann('test_y_' + name, Xtestsub)
    print 'Hallo'


def read_and_regress(feature_fn):
    Xo = read_path('project_data/train.csv')
    Xtest = read_path('project_data/test.csv')
    Xval = read_path('project_data/validate.csv')
    print 'data points: ', len(Xo)

    XM = np.matrix(Xo)
    Xval = np.matrix(Xval)
    Xtest = np.matrix(Xtest)
    means = [np.mean(XM[:,i]) for i in range(np.shape(XM)[1])]
    print 'means: ', means
    stds = [np.std(XM[:,i]) for i in range(np.shape(XM)[1])]
    print 'stds: ', stds

    Y = np.genfromtxt('project_data/train_y.csv', delimiter=',')

    Xo = XM
    Y = np.array(Y)
    print 'shape of train-data: ',Xo.shape
    print 'shape of train-label: ',Y.shape

    #Xo = Xo[0:5000,:]
    #Y = Y[0:5000]

    Xo = normalize(Xo, means, stds)
    Xtest = normalize(Xtest, means, stds)
    Xval = normalize(Xval, means, stds)
    print 'X data normalized'
    #X = principalComponents(Xo, Y)

    regress(DPGMM_classifier, 'DPGMM', Xo, Y, Xval, Xtest)


if __name__ == "__main__":
    read_and_regress(lambda x, means, stds: ortho([simple_implementation], x, means, stds))
