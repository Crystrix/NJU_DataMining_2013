function ent=Entropy(train_label)
% Entropy is the sum of p(x)log(p(x)) across all the different possible
% results
    log2=@(x) log(x)/log(2);
    results=UniqueCount(train_label);
    % Now calculate the entropy
    ent=0.0;
    for ii=1:size(results,1)
        p=results(ii,2)/size(train_label,1);
        ent=ent-p*log2(p);
    end
end