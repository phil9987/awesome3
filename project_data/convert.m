train_data = hdf5read('train.h5','/data');
train_label = hdf5read('train.h5','/label');

hdf5write('train_double_train.h5','/data',double(train_data(:,    1:30000)),'/label',double(train_label(:,    1:30000)));
hdf5write('train_double_test.h5', '/data',double(train_data(:,30000:40000)),'/label',double(train_label(:,30000:40000)));

hdf5write('test_double.h5','/data',double(hdf5read('test.h5','/data')));
hdf5write('validate_double.h5','/data',double(hdf5read('validate.h5','/data')));