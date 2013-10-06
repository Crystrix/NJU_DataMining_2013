function imp=GiniImpurity(train_label,max_unique)
    total=size(train_label,1);
    counts=UniqueCount(train_label,max_unique);
    imp=0;
    for k1=1:size(counts,1)
        p1=counts(k1,2)/total;
        for k2=1:size(counts,1)
            if k1==k2
                continue;
            else
                p2=counts(k2,2)/total;
                imp=imp+p1*p2;
            end
        end
    end
end