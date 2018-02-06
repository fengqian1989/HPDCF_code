function [est_label, scoremat] = test(obj, input)
%TEST Training function for LIBSVM (using dual formulation)
%   Refer to GenericSVM for interface definition

% ensure a model has been trained
if isempty(obj.model)
  error('A SVM model has yet to be trained');
end

scoremat = obj.model.w' * [input; ones(1,size(input,2))];
[~, est_label] = max(scoremat, [], 1);

end

