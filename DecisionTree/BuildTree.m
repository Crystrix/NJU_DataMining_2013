function node=BuildTree(train_data, train_label, is_classfication)
	if(is_classfication == 1)
        scoref=@GiniImpurity;
    else
        scoref=@RSS;
	end
	current_score=scoref(train_label);
	best_gain=0.0;
	best_criteria={};
    best_sets={};
    tic
	for column_index=1:size(train_data,2)
		column_values=sort(unique(train_data(:,column_index)));
		for ii=1:size(column_values,1)-1
            value=(column_values(ii)+column_values(ii+1))/2;
			set=DivideTree(train_data,train_label,column_index,value);
			if(size(set{1,1}.data,1)>0 && size(set{1,2}.data,1)>0)
				p=size(set{1,1}.data,1)/size(train_data,1);
				gain=current_score-p*scoref(set{1,1}.label)-(1-p)*scoref(set{1,2}.label);
				if(gain>best_gain)
                    best_gain=gain;
					best_criteria={column_index,value};
					best_sets=set;
				end
            end
        end
    end
    toc
	if best_gain >0
        leftBranch=BuildTree(best_sets{1,1}.data,best_sets{1,1}.label,is_classfication);
        rightBranch=BuildTree(best_sets{1,2}.data,best_sets{1,2}.label,is_classfication);
        node=TreeNode(best_criteria{1},best_criteria{2},{},leftBranch,rightBranch);
    else
        node=TreeNode(-1,{},UniqueCount(train_label));
    end
end