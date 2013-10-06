function result=UniqueCount(train_label,max_unique)
% Count unique elements' amount
    result=zeros(max_unique,2);
    for ii=1:size(train_label,1)
           result(train_label(ii),2)=result(train_label(ii),2)+1;
    end
    for ii=max_unique:-1:1
        if(result(ii,2)==0)
            result(ii,:)=[];
        else
            result(ii,1)=ii;    
        end
    end
end