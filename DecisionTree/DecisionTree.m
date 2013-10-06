function test_pre=DecisionTree(train_data,train_label,test_data,feat_type)
% INPUTS:
% train_data: n*d matrix with n training instances and d features
% train_label: n*1 vector, labels of training data
% test_data: m*d matrix with m test instances
% feat_type: 1*d vector, feat_type(i)=0 if the i-th feature is numerical and 1 if it is discrete

% OUTPUTS:
% test_pre: m*1 vector, the predictions of the test data

	tree=BuildTree(train_data,train_label);
    test_pre=PredictSingleData(tree,test_data);
end

