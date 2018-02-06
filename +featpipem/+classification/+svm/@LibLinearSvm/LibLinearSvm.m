classdef LibLinearSvm < handle & featpipem.classification.svm.LinearSvm
  %LIBLINEARSVM Train an SVM classifier using the LIBLINEAR library
  %
  
  properties
    % svm parameters
    c            % SVM C parameter
    bias_mul     % SVM bias multiplier
    s            % Lib-Linear objectives
  end
  
  properties(SetAccess=protected)
  end
  
  methods
    function obj = LibLinearSvm(varargin)
      obj.c = 10;
      obj.bias_mul = 1;
      obj.s = 1;
      featpipem.utility.set_class_properties(obj, varargin);
      
      obj.model = [];
    end
    train(obj, input, labels, train_samples)
    [est_label, scoremat] = test(obj, input)
    WMat = getWMat(obj)
  end
  
end

