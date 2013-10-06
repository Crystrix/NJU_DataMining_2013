function result=PredictSingleData(tree,data)
    if(~isempty(tree.results))
        result=tree.results;
    else
        current_value=data(tree.col);
        if(current_value>tree.value)
            branch=tree.tb;
        else
            branch=tree.fb;
        end
        result=PredictSingleData(branch,data);
    end
end