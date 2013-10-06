function result=Predict(tree,data)
    result=zeros(size(data,1),1);
    for ii=1:size(data,1)
        temp =PredictSingleData(tree,data(ii,:));
        [value,pos]=max(temp(:,2)); 
        result(ii)=temp(pos,1);
    end
end