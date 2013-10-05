function result=PredictSingleData(tree,data)
    if(isempty(tree.result))
        result=tree.result;
    else
        current_value=data(tree.col);
        if(current_value>tree.value)
            branch=tree.tb;
        else
            branch=tree.fb;
        end
        result=classify(observation,branch);
    end
end