function [codebook,samples,mdata] = train(obj, imlist, samples, mdata)
%TRAIN Summary of this function goes here
%   Detailed explanation goes here

% ------------------------------------------------------------------------------
% 1. Extract features for training into 'feats' matrix
%     applying any limits on number of features/images ��ȡ128*1000000��sift����
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
    %% �����feats_all�ṹΪ�����cell��Ӧ��ͬ�ʵ�ѧϰ���ڲ�cell��Ӧ��ͬ�߶ȣ�
    %% TDD�������߶��µ������ŵ�һ��Ȼ����дʵ�ѧϰ
    feats_all = cellfun(@(x) mulscalefuse(x),feats_all,'UniformOutput', false);%�ڲ�cell�ں�
    
    % if a descount limit applies, discard a fraction of features now to
    % save memory
    if obj.descount_limit > 0 % feats{ii}�ǰ������У����ԣ���㲻ͬ�ʵ�ѧϰ������
                              % Ӧ�ð���������
      feats{ii} = cellfun(@(x) vl_colsubset(x,img_descount_limit),...
          feats_all,'UniformOutput', false);
    else
      feats{ii} = cellfun(@(x) x, feats_all,'UniformOutput', false);
    end 
  end
  clear feats_all;
  % concatenate features into a single matrix
  feats = cat(2,feats{:}); % ��Ƕ��cellչ������ͬ�����Ĳ�ͬ�ʵ���������
  feats = num2cell(feats,2); % ����������ÿһ��ʵ���������Ӧ���������������һ��cell
  feats = cellfun(@(x) cat(2, x{:}), feats, 'UniformOutput', false); % ��ͬ�ʵ�����λ�ڲ�ͬ��cell

  cellfun(@(x) fprintf('%d features extracted\n', size(x,2)),...
      feats,'UniformOutput',false);% ��ͬcell��Ӧ��ʾ
  
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
% 2. Cluster codebook centres   ����codebook 128*1024 ��1024��cluster centers
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

% ����ͬ�߶ȵ������ŵ�һ��ͳһѧϰһ���ʵ�
function feaF = mulscalefuse(feaM)
% ��߶�ͬ�������
if iscell(feaM)
    feaF = cat(2, feaM{:});
else
    feaF = feaM;
end

end
