function [codebook,samples,mdata] = train(obj, imlist, samples, mdata)
%TRAIN Summary of this function goes here
%   Detailed explanation goes here

% ------------------------------------------------------------------------------
% 1. Extract features for training into 'feats' matrix
%     applying any limits on number of features/images 提取128*1000000的sift特征
% ------------------------------------------------------------------------------
if isempty(samples)
  % if trainimage_count was not left at it's default value
  % (indicating all detected images should be used for training)
  % select a subset of the images
  if obj.trainimage_limit > 0
    idxs = 1:length(imlist);
    idxs = vl_colsubset(idxs, obj.trainimage_limit);
    imlist = imlist(idxs);
  end
  
  if obj.descount_limit > 0
    % set truncation value for image features just a little bit
    % larger than descount_limit, so if there are any images
    % with fewer than descount_limit/numImages we still have
    % some chance of getting descount_limit descriptors in the end
    img_descount_limit = ceil(obj.descount_limit / ...
      length(imlist) * 1.1);
    fprintf('Extracting a maximum of %d features from each image...\n', ...
      img_descount_limit);
  end
  
  feats = cell(length(imlist),1);

  % iterate through images, computing features
  pfImcount = length(imlist);
%  for ii = 1:length(imlist)
  parfor ii = 1:length(imlist)
    fprintf('\nComputing features for: %s %f %% complete\n', ...
      imlist{ii}, ii/pfImcount*100.00);
    
    im = obj.imloader_.process(imlist{ii}); %#ok<PFBNS>
    feats_all = obj.featextr_.compute(im); 
    
    %==========================================================================
    %% 这里的feats_all结构为：外层cell对应不同词典学习和内层cell对应不同尺度，
    %% TDD将各个尺度下的特征放到一起，然后进行词典学习
    feats_all = cellfun(@(x) mulscalefuse(x),feats_all,'UniformOutput', false);%内层cell融合
    
    % if a descount limit applies, discard a fraction of features now to
    % save memory
    if obj.descount_limit > 0 % feats{ii}是按列排列，所以，外层不同词典学习的特征
                              % 应该按照行排列
      feats{ii} = cellfun(@(x) vl_colsubset(x,img_descount_limit),...
          feats_all,'UniformOutput', false);
    else
      feats{ii} = cellfun(@(x) x, feats_all,'UniformOutput', false);
    end 
  end
  clear feats_all;
  % concatenate features into a single matrix
  feats = cat(2,feats{:}); % 将嵌套cell展开，不同样本的不同词典特征排列
  feats = num2cell(feats,2); % 将上述矩阵，每一类词典特征所对应的所有样本，组成一个cell
  feats = cellfun(@(x) cat(2, x{:}), feats, 'UniformOutput', false); % 不同词典特征位于不同的cell

  cellfun(@(x) fprintf('%d features extracted\n', size(x,2)),...
      feats,'UniformOutput',false);% 不同cell对应显示
  
  if obj.descount_limit > 0
    % select subset of features for training
    feats = cellfun(@(x) vl_colsubset(x,obj.descount_limit),...
        feats,'UniformOutput', false);
    % output status message
    cellfun(@(x) fprintf(['%d features will be used for training of codebook'...
        '(%f %%)\n'],obj.descount_limit, obj.descount_limit/size(x,2)*100.0),...
        feats,'UniformOutput',false);
  end
  samples = cellfun(@(x) single(x), feats,'UniformOutput',false);
  %============================================================================
else
  feats = samples;
end

% ----------------------------------------------------------------------------
% 2. Cluster codebook centres   构建codebook 128*1024 即1024个cluster centers
% ----------------------------------------------------------------------------

fprintf('Clustering features...\n');

% if maxcomps is below 1, then use exact kmeans, else use approximate
% kmeans with maxcomps number of comparisons for distances
if obj.maxcomps < 1
  codebook = cellfun(@(x) vl_kmeans(x, obj.cluster_count, ...
    'verbose', 'algorithm', 'elkan'), feats,'UniformOutput', false);
else
  codebook = cellfun(@(x) featpipem.lib.annkmeans(x, obj.cluster_count, ...
    'verbose', true, 'MaxNumComparisons', obj.maxcomps, 'MaxNumIterations',...
    150), feats,'UniformOutput', false);
end

fprintf('Done training codebook!\n');

end

% 将不同尺度的特征放到一起，统一学习一个词典
function feaF = mulscalefuse(feaM)
% 多尺度同处理策略
if iscell(feaM)
    feaF = cat(2, feaM{:});
else
    feaF = feaM;
end

end
