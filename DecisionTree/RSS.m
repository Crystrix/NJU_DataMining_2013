function result=RSS(train_label)
% Entropy is the sum of p(x)log(p(x)) across all the different possible
% results
    m=mean(train_label);
    result=0;
    for ii=1:size(train_label,1)
        result=result+(train_label(ii)-m)^2;
    end
end