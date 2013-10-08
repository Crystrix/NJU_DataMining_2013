function imp=GiniImpurity2(train_label)
    total=size(train_label,1);
    counts=UniqueCount2(train_label);
    p=(counts(:,2)/total);
    imp=1-sum(p'.^2,2);
end