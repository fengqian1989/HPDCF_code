function [assign,poolpos] = encode(obj, feats)
%ENCODE Encode features using the VLAD method (hard assignment)
% feats outer cell ----------- different dictionary
% feats inter cell ----------- different scales features

% Apply encoding ------------------------------------------------------
% corresponding outer cell
[assign,poolpos] = cellfun(@(x,y,z) mulencode(obj,x,y,z), obj.kdtree_,...
    obj.codebook_, feats,'UniformOutput',false);

end

%%
function [assign,codeids] = mulencode(obj, kdtree, codebook, mulfeats)

% Apply encoding towards each scale features ---------------------------
% corresponding to inter cell
knn = obj.num_nn;

if obj.max_comps ~= -1
  % using ann...
  codeids = cellfun(@(x) vl_kdtreequery(kdtree, codebook, x,...
    'MaxComparisons', obj.max_comps, 'NumNeighbors', knn), mulfeats,...
    'UniformOutput',false);
else
  % using exact assignment...
  [~,codeids] = cellfun(@(x) featpipem.utility.max_k(...
      -vl_alldist(codebook, x), knn),mulfeats,'UniformOutput',false);
end

assign = cellfun(@(x,y) knn_assign(x,y,codebook,knn), mulfeats,codeids,...
    'UniformOutput',false);
codeids = cellfun(@(x) int32(x(1:knn,:)), codeids, 'UniformOutput',false);

end


% knn assignment -- VLAD-k
function assign = knn_assign(feats,codeids,codebook,knn)
% coresponding to single feature block
assign = cell(knn, 1);
for i = 1:knn
    assign{i} = feats - codebook(:,codeids(i,:));
end
assign = cell2mat(assign);

end






