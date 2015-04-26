train_data = hdf5read('train.h5','/data');
train_label = hdf5read('train.h5','/label');

n = size(train_data,2);
perm = randperm(n);
train_data = double(train_data(:,perm));
train_label = double(train_label(:,perm));
train_label_one_hot = ind2vec(train_label+1);
s = 0.9*n;

hdf5write('train_double_train.h5','/data',train_data(:,1:s),      '/label',train_label(:,1:s));
hdf5write('train_double_test.h5', '/data',train_data(:,(s+1):end),'/label',train_label(:,(s+1):end));

validate_data = double(hdf5read('validate.h5','/data'));
hdf5write('validate_double.h5','/data',validate_data);

test_data = double(hdf5read('test.h5','/data'));
hdf5write('test_double.h5','/data',test_data);