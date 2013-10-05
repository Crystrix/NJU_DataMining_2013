function result=Predict(tree,data)
    result=zeros(size(data,1));
    for i=1:size(data,1)
        result(i)=PredictSingleData(tree,data);
    end
end