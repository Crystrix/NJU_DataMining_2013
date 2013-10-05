function result=UniqueCount(train_label)
% Count unique elements' amount
	unique_elements=unique(train_label);
    unique_elements(:,2)=zeros(length(unique_elements),1);
    for ii=1:size(unique_elements,1)
        unique_elements(ii,2)=length(find(train_label==unique_elements(ii,1)));
    end
	result=unique_elements;
end