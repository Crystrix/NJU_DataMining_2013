function node=BuildTree(train_data, train_label, is_classfication)
	if(is_classfication == 0)
        scoref=@Entropy;
    end
	current_score=Entropy(train_label);
	best_gain=0.0;
	best_criteria={};
    best_sets={};
	for column_index=1:size(train_data,2)
		column_values=sortrows(unique(train_data(:,column_index)),-1);
		for ii=1:size(column_values,1)
			set=DivideTree(train_data,train_label,column_index,column_values(ii));
			if(size(set{1,1}.data,1)>0 && size(set{1,2}.data,1)>0)
				p=size(set{1,1}.data,1)/size(train_data,1);
				gain=current_score-p*scoref(set{1,1}.label)-(1-p)*scoref(set{1,2}.label);
				if(gain>best_gain)
                    best_gain=gain;
					best_criteria={column_index,column_values(ii)};
					best_sets=set;
				end
			end
		end
	end
	if best_gain >0
        leftBranch=BuildTree(best_sets{1,1}.data,best_sets{1,1}.label);
        rightBranch=BuildTree(best_sets{1,2}.data,best_sets{1,2}.label);
        node=TreeNode(best_criteria{1},best_criteria{2},{},leftBranch,rightBranch);
    else
        node=TreeNode(-1,{},UniqueCount(train_label));
	end
	
end