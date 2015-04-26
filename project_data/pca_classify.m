%convert;

%COEFF_ALL = pca(train_data');
COEFF = COEFF_ALL(:,1:200);
labels = train_label';

data_p = train_data' * COEFF;
[trainInd,valInd,testInd] = dividerand(size(data_p,1),0.8,0,0.2);

classifier = fitctree(data_p(trainInd,:), labels(trainInd,:));
%classifier = fitensemble(data_p(trainInd,:), labels(trainInd,:));

%classifier = fitensemble(50,data_p(trainInd,:),labels(trainInd,:),'OOBPred','On','Method','classification');

score_train = sum(predict(classifier,data_p(trainInd,:)) == labels(trainInd,:))/(0.8*size(data_p,1))
score_test  = sum(predict(classifier,data_p(testInd,:)) == labels(testInd,:))/(0.2*size(data_p,1))

data_val = validate_data' * COEFF;

data_val_labels = predict(classifier,data_val);
csvwrite('validate_matlab_output.txt',data_val_labels);