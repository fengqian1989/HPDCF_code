classdef Constant
  %CONSTANT Summary of this class goes here
  %   Detailed explanation goes here
  
  properties(Constant=true)
    POOL_MAX = 1
    POOL_SUM = 2
    SVMKER_NONE = 1
    SVMKER_HELLINGER = 2
    SVMKER_KCHI2 = 3
  end
  
  methods(Static)
    function data = parse_pool_type(str)
      if strcmp(str, 'sum')
        data = featpipem.Constant.POOL_SUM;
      elseif strcmp(str, 'max')
        data = featpipem.Constant.POOL_MAX;
      else
        error('unknown pool type: %s', str);
      end
    end
    function data = parse_svmker_type(str)
      if strcmp(str, 'none')
        data = featpipem.Constant.SVMKER_NONE;
      elseif strcmp(str, 'hellinger')
        data = featpipem.Constant.SVMKER_HELLINGER;
      elseif strcmp(str, 'kchi2')
        data = featpipem.Constant.SVMKER_KCHI2;
      else
        error('unknown svm kernel type: %s', str);
      end
    end
  end
  
end

