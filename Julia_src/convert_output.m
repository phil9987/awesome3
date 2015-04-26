validate_classes = double(hdf5read('validate_output.h5','/label'))';
[M, validate_label] = max(validate_classes,[],2);
csvwrite('validate_output.txt',validate_label-1);

test_classes = double(hdf5read('test_output.h5','/label'))';
[M, test_label] = max(test_classes,[],2);
csvwrite('test_output.txt',test_label-1);