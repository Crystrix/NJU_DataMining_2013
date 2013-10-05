function result=DivideTree(train_data,train_label,column_index,value)
    result=cell(1,2);
    result{1,1}.label = [];
    result{1,1}.data = [];
	result{1,2}.label = [];
    result{1,2}.data = [];
	split_function=@(row_index) train_data(row_index,column_index)>=value;
	for ii=1:size(train_data,1)
		if(split_function(ii))
            result{1,1}.label=[result{1,1}.label;train_label(ii,:)];
			result{1,1}.data=[result{1,1}.data;train_data(ii,:)];
        else
            result{1,2}.label=[result{1,2}.label;train_label(ii,:)];
			result{1,2}.data=[result{1,2}.data;train_data(ii,:)];
		end
	end
end