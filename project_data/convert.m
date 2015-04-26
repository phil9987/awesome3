train_data = hdf5read('train.h5','/data');
train_label = hdf5read('train.h5','/label');

n = size(train_data,2)
perm = randperm(n);
train_data = train_data(:,perm);
s = 0.8*n;
hdf5write('train_double_train.h5','/data',double(train_data(:,1:s)),        '/label',double(train_label(:,1:s)));
hdf5write('train_double_test.h5', '/data',double(train_data(:,(s+1):40000)),'/label',double(train_label(:,(s+1):40000)));

hdf5write('test_double.h5','/data',double(hdf5read('test.h5','/data')));
hdf5write('validate_double.h5','/data',double(hdf5read('validate.h5','/data')));