function node=BuildTree(train_data, train_label, is_classfication,max_unique)
	if(is_classfication == 1)
        scoref=@GiniImpurity;
    else
        scoref=@RSS;
	end
	current_score=scoref(train_label,max_unique);
	best_gain=0.0;
	best_criteria={};
    %best_sets={};
	for column_index=1:size(train_data,2)
		column_values=sort(unique(train_data(:,column_index)));
		for ii=1:size(column_values,1)-1
            value=(column_values(ii)+column_values(ii+1))/2;
            index=train_data(:,column_index) < value;
			left_label=train_label(index);
            right_label=train_label(~index);
			if(size(left_label,1)>0 && size(left_label,1)>0)
				p=size(right_label,1)/size(train_data,1);
				gain=current_score-p*scoref(right_label,max_unique)-(1-p)*scoref(left_label,max_unique);
				if(gain>best_gain)
                    best_gain=gain;
					best_criteria={column_index,value};
					%best_sets=set;
				end
            end
        end
    end
	if best_gain >0
        best_sets=DivideTree(train_data,train_label,best_criteria{1},best_criteria{2});
        leftBranch=BuildTree(best_sets{1,1}.data,best_sets{1,1}.label,is_classfication,max_unique);
        rightBranch=BuildTree(best_sets{1,2}.data,best_sets{1,2}.label,is_classfication,max_unique);
        node=TreeNode(best_criteria{1},best_criteria{2},{},leftBranch,rightBranch);
    else
        node=TreeNode(-1,{},UniqueCount(train_label,max_unique));
    end
end