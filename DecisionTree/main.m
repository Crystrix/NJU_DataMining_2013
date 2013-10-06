function main
name='segment'; % data set name

%load data
data=load([name,'.data']);
feat_type=data(1,:);
label=data(2:end,end);
data=data(2:end,1:end-1);
n=size(data,1);

%load index
idx=load('permutation.txt');
idx=idx+1;
idx(idx>n)=[];
nFold=floor(n/10);

% 10 folds cross validation
result=zeros(1,10);
for i=1:10
    i
    % data partition
    idx_te=idx((i-1)*nFold+1:min(n,i*nFold));
    idx_tr=1:n;
    idx_tr(idx_te)=[];
    train_data=data(idx_tr,:);
    train_label=label(idx_tr);
    test_data=data(idx_te,:);
    test_label=label(idx_te);
    
    test_pre=DecisionTree(train_data,train_label,test_data,feat_type);
    %test_pre=classregtree(train_data,train_label,'prune','off','method','classification','minparent',1);
    %test_pre=eval(test_pre,test_data);
    %test_pre=cell2mat(test_pre);
    %test_pre=str2num(test_pre);
    if(strcmp(name,'housing')||strcmp(name,'meta'))
        result(i)=sqrt(sum((test_pre-test_label).^2)/size(test_label,1));
    else
        result(i)=sum(test_pre==test_label)/size(test_label,1);
    end
end
disp(['the average accuracy (RMSE) of 10 folds CV is ',num2str(mean(result))]);
end
