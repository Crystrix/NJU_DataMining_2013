classdef DecisionTreeClass
%CLASSREGTREE Create a classification and regression tree object.
    properties
        node = zeros(0,1); %节点标记
        parent = zeros(0,1); %对应节点的父节点
        var = zeros(0,1); %每个节点划分对应的属性下标
        cut = zeros(0,1); %每个节点划分对应的值
        class = zeros(0,1);
        children = zeros(0,2); %每个节点的子节点
        nodeprob = zeros(0,1);
        nodeerr = zeros(0,1);
        varassoc = cell(0,1);
        surrvar = cell(0,1);
        surrcut = cell(0,1);
        surrflip = cell(0,1);
        prunelist = zeros(0,1);
        alpha = [];
        ntermnodes = [];
    end
    methods
        function result = DecisionTreeClass(x,y,if_classification,if_surrogate)
            result = treefit(result,x,y,if_classification,if_surrogate);
            result=getpruneinfo(result);
            threshlod = 5;
            level = find(result.alpha<=threshlod,1,'last') - 1;
            result = subtree(result,level);               % remove stuff using optimal sequence

        end 
        function result = Predict(obj,data)
            result=zeros(size(data,1),1);
            for ii=1:size(data,1)
                temp =PredictSingleData(obj,data(ii,:));
                result(ii)=temp;
            end
        end
    end
end % classdef
function Tree = subtree(Tree,p)
%SUBTREE Get subtree from tree indexed by pruning point.

whenpruned = Tree.prunelist;
v = find(whenpruned>0 & whenpruned<=p);
if ~isempty(v)
   Tree = prunenodes(Tree,v);
end
end
function Tree=treefit(Tree,X,Y,if_classification,if_surrogate)
    t = isnan(Y);
    if any(t)
        X(t,:) = [];
        Y(t) = [];
    end
    t = all(isnan(X),2);
    if any(t)
        X(t,:) = [];
        Y(t) = [];
    end
    [N,nvars] = size(X);
    minleaf=1;
    minparent=1;
    row_number = size(X,1); %The length of samples
    max_space = 2*ceil(N/minleaf)-1; %Pre-perpared max possible space
    nodenumber = zeros(max_space,1);
    nodeprob = zeros(max_space,1);
    parent = zeros(max_space,1);
    yfitnode = zeros(max_space,1);
    cutvar = zeros(max_space,1);
    cutpoint = zeros(max_space,1);
    children = zeros(max_space,2);
    max_unique = max(Y);
    resuberr = zeros(max_space,1);

    
    
	if(if_classification == 1)
        scoref=@GiniImpurity2;
    else
        scoref=@RSS;
    end
    
	if(if_surrogate == 1)
       varassoc = cell(max_space,1);
       surrvar = cell(max_space,1);
       surrcut = cell(max_space,1);
       surrflip = cell(max_space,1);
   end
    
    
    assignednode = cell(max_space,1);% list of instances assigned to this node
    assignednode{1} = 1:row_number;
    nextunusednode = 2;
    nodenumber(1) = 1;

    tnode = 1;
    while(tnode < nextunusednode)
        nodeprob(tnode) = length(assignednode{tnode});
        noderows = assignednode{tnode};
        Nt = length(noderows);
        Ynode = Y(noderows,:);
        bestcrit = -Inf;


        if (Nt>=minparent)
            Xnode = X(noderows,:);
            bestvar = 0;
            bestcut = 0;
            x = Xnode;
            current_score=scoref(Ynode);
            if(if_classification)
                temp = UniqueCount2(Ynode);
                [maximum, pos] = max(temp(:,2));
                yfitnode(tnode) = temp(pos,1);
                resuberr(tnode)=maximum/sum(temp(:,2));
            else
                yfitnode(tnode) = mean(Ynode);
                resuberr(tnode)= RSS(Ynode);
            end
            
            for column_index=1:nvars
                column_values=sort(unique(x(:,column_index)));
                for ii=1:size(column_values,1)-1
                    value=(column_values(ii)+column_values(ii+1))/2;
                    left_label=Ynode(x(:,column_index) < value);
                    right_label=Ynode(x(:,column_index) >= value);
                    if(size(left_label,1)>0 && size(left_label,1)>0)
                        p=size(right_label,1)/(length(left_label)+length(right_label));
                        gain=current_score-p*scoref(right_label)-(1-p)*scoref(left_label);
                        if(gain>bestcrit)
                            bestcrit=gain;
                            bestvar=column_index;
                            bestcut=value;
                        end
                    end
                end
            end
            
            if((if_classification== 1 && bestcrit>0) || bestcrit > 1)
                %nvarsplit(bestvar) = nvarsplit(bestvar)+1;
                x = Xnode(:,bestvar);
                cutvar(tnode) = bestvar;
                leftside = x<bestcut;
                rightside = x>=bestcut;

                % Store split position, children, parent, and node number
                cutpoint(tnode) = bestcut;
                children(tnode,:) = nextunusednode + (0:1);
                nodenumber(nextunusednode+(0:1)) = nextunusednode+(0:1)';
                parent(nextunusednode+(0:1)) = tnode;
                if if_surrogate
                     [tvarassoc,tsurrvar,tsurrcut,tsurrflip,tleftORright] = ...
                         findsurrogate(Xnode,nvars,bestvar,leftside,rightside);
                                  % Sort vars by their associations with the best var.
                     [~,idxvarsort] = sort(tvarassoc,'descend');

                     % Store surrogate cuts and if they need to be flipped
                     surrcut(tnode) = {tsurrcut(idxvarsort)};
                     surrflip(tnode) = {tsurrflip(idxvarsort)};

                     % Store variables for surrogate splits.
                     % For categorical vars, store negative indices.
                     tsurrvar = tsurrvar(idxvarsort);
                     surrvar(tnode) = {tsurrvar};

                     % Store variable associations
                     varassoc(tnode) = {tvarassoc(idxvarsort)};

                     % Append lists of observations to be assigned to left and
                     % right children
                     for jmis=1:length(idxvarsort)
                         idxmissing = (1:Nt)';
                         idxmissing = idxmissing(~(leftside | rightside));
                         if isempty(idxmissing)
                             break;
                         else
                             surrmissing = tleftORright(idxmissing,idxvarsort(jmis));
                             leftside(idxmissing(surrmissing<0)) = true;
                             rightside(idxmissing(surrmissing>0)) = true;
                         end
                     end 
                end
                % Assign observations for the next node
                assignednode{nextunusednode} = noderows(leftside);
                assignednode{nextunusednode+1} = noderows(rightside);
                % Update next node index
                nextunusednode = nextunusednode+2;
            end
        end
        tnode = tnode + 1;
        
    end
    
    topnode        = nextunusednode - 1;
    Tree.node      = nodenumber(1:topnode);
    Tree.parent    = parent(1:topnode);
    Tree.class     = yfitnode(1:topnode);
    Tree.var       = cutvar(1:topnode);
    Tree.cut       = cutpoint(1:topnode);
    Tree.children  = children(1:topnode,:);
    Tree.nodeprob  = nodeprob(1:topnode,:);
    Tree.nodeerr   = resuberr(1:topnode);

    if ~isempty(surrvar)
        Tree.varassoc = varassoc(1:topnode);
        Tree.surrvar = surrvar(1:topnode);
        Tree.surrcut = surrcut(1:topnode);
        Tree.surrflip = surrflip(1:topnode);
    end
end


function result=PredictSingleData(Tree, data)
    index = 1;
    while(Tree.children(index,1) ~= 0)
        column=Tree.var(index);
        if(isnan(data(column)))
            sign = 0;
            for ii=1:length(Tree.surrvar{index})
                if(~isnan(data(column)))
                    if(data(column) < Tree.surrcut{index})
                        if(Tree.tsurrflip{index} == 1)
                            index = Tree.children(index,1);
                        else
                            index = Tree.children(index,2);
                        end
                    else
                        if(Tree.tsurrflip{index} == 1)
                            index = Tree.children(index,2);
                        else
                            index = Tree.children(index,1);
                        end
                    end
                    sign = 1;
                    break;
                end
            end
            if(sign == 0)
                if(Tree.nodeprob(Tree.children(index,1)) > Tree.nodeprob(Tree.children(index,2)))
                    index = Tree.children(index,1);
                else
                    index = Tree.children(index,2);
                end
            end
        elseif(data(column) < Tree.cut(index))
            index = Tree.children(index,1);
        else
            index = Tree.children(index,2);
        end
    end
    result = Tree.class(index);
end

function r=risk(t,j,if_split)
    if nargin<2
        j = 1:numnodes(t);
    else
        j = j(:);
    end
    % Compute appropriate node probability
    if     if_split  % regular risk
        tprob = t.nodeprob(j);
    elseif ~if_split % risk due to unsplit data
        tprob = nodeprob(t,j,0);
    end
        r = tprob.*t.nodeerr(j);
end
    

% ------------------------------------------------------------------
function Tree=getpruneinfo(Tree)
    %GETPRUNEINFO Get optimal pruning information and store into decision tree.

    % Start from the smallest tree that minimizes R(alpha,T) for alpha=0
    N = length(Tree.node);
    parent     = Tree.parent;
    children   = Tree.children;

    isleaf = Tree.var(:)==0;
    nleaves = sum(isleaf);
    adjfactor = 1 + 100*eps;

    % Work up from the bottom of the tree to compute, for each branch node,
    % the number of leaves under it and sum of their costs
    treatasleaf = isleaf';
    nodecost = risk(Tree,1:N,1);
    unsplitcost = risk(Tree,1:N,0);
    costsum = nodecost;
    nodecount = double(isleaf);
    while(true)
       % Find ''twigs'' which I define as branches with two leaf children
       branches = find(~treatasleaf);
       twig = branches(sum(treatasleaf(children(branches,:)),2) == 2);
       if isempty(twig), break; end;    % worked our way up to the root node

       % Add the costs and sizes of the two children, give to twig
       kids = children(twig,:);
       costsum(twig)   = unsplitcost(twig) + sum(costsum(kids'),1)';
       nodecount(twig) = sum(nodecount(kids'),1)';
       treatasleaf(twig) = 1;
    end

    % Now start pruning to generate a sequence of smaller trees
    whenpruned = zeros(N,1);
    branches = find(~isleaf);
    prunestep = 0;
    allalpha = zeros(N,1);
    ntermnodes = zeros(N,1);
    ntermnodes(1) = nleaves;
    while(~isempty(branches))
       prunestep = prunestep + 1;

       % Compute complexity parameter -- best tree minimizes cost+alpha*treesize
       alpha = max(0, nodecost(branches) - costsum(branches)) ./ ...
               max(eps,nodecount(branches) - 1);
       bestalpha = min(alpha);
       toprune = branches(alpha <= bestalpha*adjfactor);

       % Mark nodes below here as no longer on tree
       wasleaf = isleaf;
       kids = toprune;
       while ~isempty(kids)
          kids = children(kids,:);
          kids = kids(kids>0);
          kids(isleaf(kids)) = [];
          isleaf(kids) = 1;
       end
       newleaves = toprune(~isleaf(toprune));
       isleaf(toprune) = 1;

       % Remember when branch was pruned, also perhaps leaves under it
       whenpruned(isleaf~=wasleaf & whenpruned==0) = prunestep;
       whenpruned(toprune) = prunestep;   % this branch was pruned

       % Update costs and node counts
       for j=1:length(newleaves)          % loop over branches that are now leaves
          node = newleaves(j);
          diffcost  = nodecost(node) - costsum(node);
          diffcount = nodecount(node) - 1;
          while(node~=0)                  % work from leaf up to root
             nodecount(node) = nodecount(node) - diffcount;
             costsum(node)   = costsum(node) + diffcost;
             node = parent(node);         % move to parent node
          end
       end

       allalpha(prunestep+1) = bestalpha;
       ntermnodes(prunestep+1) = nodecount(1);

       % Get list of branches on newly pruned tree
       branches = find(~isleaf);
end

Tree.prunelist  = whenpruned;
Tree.alpha      = allalpha(1:prunestep+1);
Tree.ntermnodes = ntermnodes(1:prunestep+1);

end

function p=nodeprob(t,j,if_split)
%NODEPROB Node probability.
%   P=NODEPROB(T) returns an N-element vector P of the probabilities of the
%   nodes in the tree T, where N is the number of nodes.  The probability
%   of a node is computed as the proportion of observations from the
%   original data that satisfy the conditions for the node.  For a
%   classification tree, this proportion is adjusted for any prior
%   probabilities assigned to each class.
%
%   P=NODEPROB(T,J) takes an array J of node numbers and returns the 
%   probabilities for the specified nodes.
%
%   See also CLASSREGTREE, CLASSREGTREE/NUMNODES, CLASSREGTREE/NODESIZE.

%   Copyright 2006-2007 The MathWorks, Inc. 
%   $Revision: 1.1.8.3 $  $Date: 2012/03/01 02:30:07 $

if     nargin<2
    p = t.nodeprob;
    return;
end
j = j(:);

if     nargin<3
    p = t.nodeprob(j);
else
    if ~if_split
        p = zeros(numel(j),1);
        kids = t.children(j,:);
        isbr = all(kids>0,2);
        if any(isbr)
            p(isbr) = t.nodeprob(j(isbr)) - sum(t.nodeprob(kids(isbr,:)'),1)';
        end
    else
        error(message('stats:classregtree:nodeprob:BadMode'));
    end
end
end
% ---------------------------------------
function [varassoc,surrvar,surrcut,surrflip,leftORright] = ...
    findsurrogate(Xnode,nvar,bestvar,leftside,rightside)
% Get number of vars and make default output
N = size(Xnode,1);
varassoc = zeros(1,nvar);
surrcut = cell(1,nvar);
surrvar = false(1,nvar);
surrflip = zeros(1,nvar);
leftORright = zeros(N,nvar);

% Left and right probabilities for the best split
pL = length(find(leftside)==1)/size(Xnode,1);
pR = length(find(rightside)==1)/size(Xnode,1);
minp = min(pL,pR);

% Loop over variables
for ivar=1:nvar
    % Get the predictor from the original data X
    jvar = ivar;
 
    % If best-split variable, assign left and right cases.
    % Best variable is not a surrogate variable but we need to compute
    % varimp for it.
    if jvar==bestvar
        leftORright(leftside,ivar)  = -1;
        leftORright(rightside,ivar) = +1;
    else
        %
        % Find the split that maximizes pLL+pRR
        %
        x = Xnode(:,jvar);        
        % Find NaN's and sort
        idxnotnan = find(~isnan(x));
        if isempty(idxnotnan)
            continue;
        end
        [x,idxsorted] = sort(x(idxnotnan));
        idx = idxnotnan(idxsorted);
            
        % Determine if there's anything to split along this variable
        maxeps = max(eps(x(1)), eps(x(end)));
        if x(1)+maxeps > x(end)
            continue;
        end
            
        % Accept only splits on rows with distinct values
        idxdistinct = find(x(1:end-1) + ...
            max([eps(x(1:end-1)) eps(x(2:end))],[],2) < x(2:end));
        if isempty(idxdistinct)
            continue;
        end
        idxdistinct(end+1) = length(x);
        x = x(idxdistinct);
        w = ones(size(Xnode,1),2);
        w(rightside(idx),1) = 0;
        w(leftside(idx),2) = 0;
        w = cumsum(w,1);
        w = w(idxdistinct,:);
        w = w/size(Xnode,1);
            % Find split maximizing pLL+pRR
        [wLLandRRmax,i1] = ...
            max(w(1:end-1,1)+w(end,2)-w(1:end-1,2));
        [wLRandRLmax,i2] = ...
            max(w(end,1)-w(1:end-1,1)+w(1:end-1,2));
        if wLLandRRmax<wLRandRLmax
            surrflip(ivar) = -1;
            pLL = w(end,1)-w(i2,1);
            pRR = w(i2,2);
            cut = 0.5*(x(i2)+x(i2+1));
        else
            surrflip(ivar) = +1;
            pLL = w(i1,1);
            pRR = w(end,2)-w(i1,2);
            cut = 0.5*(x(i1)+x(i1+1));
        end
        x = Xnode(:,jvar);
        leftORright(x<cut,ivar)  = -surrflip(ivar);
        leftORright(x>=cut,ivar) = +surrflip(ivar);
       
        % Get association
        if minp>1-pLL-pRR && pLL>0 && pRR>0
            surrvar(ivar) = true;
            surrcut{ivar} = cut;
            varassoc(ivar) = (minp-(1-pLL-pRR)) / minp;
        end
    end
end

% Return only values for surrogate split vars (satisfying varassoc>0).
% varimp is the only exception - it keeps values for all variables.
varassoc = varassoc(surrvar);
surrcut = surrcut(surrvar);
surrflip = surrflip(surrvar);
leftORright = leftORright(:,surrvar);
surrvar = find(surrvar==1);
end

function Tree = prunenodes(Tree,branches)
%PRUNENODES Prune selected branch nodes from tree.

N = length(Tree.node);

% Find children of these branches and remove them
parents = branches;
tokeep = true(N,1);
kids = [];
while(true)
   newkids = Tree.children(parents,:);
   newkids = newkids(:);
   newkids = newkids(newkids>0 & ~ismember(newkids,kids));
   if isempty(newkids), break; end
   kids = [kids; newkids];
   tokeep(newkids) = false;
   parents = newkids;
end

% Convert branches to leaves by removing split rule and children
Tree.var(branches) = 0;
Tree.cut(branches) = 0;
Tree.children(branches,:) = 0;

% Get new node numbers from old node numbers
ntokeep = sum(tokeep);
nodenums = zeros(N,1);
nodenums(tokeep) = (1:ntokeep)';

% Reduce tree, update node numbers, update child/parent numbers
Tree.parent    = Tree.parent(tokeep);
Tree.class     = Tree.class(tokeep);
Tree.var       = Tree.var(tokeep);
Tree.cut       = Tree.cut(tokeep);
Tree.children  = Tree.children(tokeep,:);
Tree.nodeprob  = Tree.nodeprob(tokeep);
Tree.nodeerr   = Tree.nodeerr(tokeep);
Tree.node      = (1:ntokeep)';
mask = Tree.parent>0;
Tree.parent(mask) = nodenums(Tree.parent(mask));
mask = Tree.children>0;
Tree.children(mask) = nodenums(Tree.children(mask));

if ~isempty(Tree.surrvar)
    Tree.varassoc = Tree.varassoc(tokeep);
    Tree.surrvar = Tree.surrvar(tokeep);
    Tree.surrcut = Tree.surrcut(tokeep);
    Tree.surrflip = Tree.surrflip(tokeep);
    isleaf = Tree.var==0;
    Tree.varassoc(isleaf) = {[]};
    Tree.surrvar(isleaf) = {[]};
    Tree.surrcut(isleaf) = {{}};
    Tree.surrflip(isleaf) = {[]};
end
end
