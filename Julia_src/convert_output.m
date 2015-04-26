validate_classes = double(hdf5read('validate_output.h5','/label'))';
[M, validate_label] = max(validate_classes,[],2);
csvwrite('validate_output.txt',validate_label-1);

% TODO: Same for test