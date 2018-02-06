classdef GenericFeatExtractor < handle
  %GENERICFEATEXTRACTOR Generic interface for extracting image features
  
  properties
    norm_type
    out_dim
  end
  
  properties(SetAccess = protected)
  end
  
  methods(Abstract)
    [feats, frames] = compute(obj, im)
  end
  
  methods
    function feats = post_process(obj, feats)
      if strcmp(obj.norm_type, 'l2')
%         feats = feats ./ repmat( max( eps, sqrt( sum(feats.^2, 1) ) ), [size(feats, 1) 1] );
        mulfeats = cell(length(feats),1);
        for fi = 1: length(feats)
            mulfeats{fi} = cellfun(@(x) x./repmat(max(eps,sqrt(sum(x.^2,1))),...
                [size(x,1) 1]),feats{fi},'UniformOutput', false);
        end
        feats = mulfeats;
      end
    end
  end
  
end

