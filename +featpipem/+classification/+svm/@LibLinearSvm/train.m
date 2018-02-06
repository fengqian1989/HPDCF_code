function train(obj, input, labels, train_samples)
%TRAIN Testing function for LIBLINEARSVM
%   Refer to GenericSVM for interface definition

% input is of dimensions feat_dim x feat_count
% i.e. column features

% convenience variables
num_classes = length(labels);
feat_dim = size(input,1);
feat_count = size(input,2);

% ensure input is of correct form
if exist('train_samples','var')
  input = input(:,train_samples);
end
if ~issparse(input)
  input = sparse(double(input));
end

% prepare temporary output model storage variables
libsvm = cell(1, num_classes);
libsvm_flipscore = zeros(1, num_classes);

% train models for each class in turn
svm_option = sprintf('-s %d -c %f -B 1', obj.s, obj.c);
for ci = 1:num_classes
  labels_cls = -ones(feat_count,1);
  labels_cls(labels{ci}) = 1;
  labels_cls = labels_cls(train_samples);
  
  libsvm{ci} = train(labels_cls, input, svm_option, 'col');
  % in two-category classification, the first label encountered is
  % assigned as +1, so if the opposite is true in the label set,
  % set a flag in the libsvm struct to indicate this
  libsvm_flipscore(ci) = (labels_cls(1) == -1);
end

% copy across trained model
obj.model = struct;
obj.model.libsvm = libsvm;
obj.model.libsvm_flipscore = libsvm_flipscore;
obj.model.w = zeros(feat_dim+1, num_classes, 'single');
for i = 1:num_classes
  if ~libsvm_flipscore(i)
    obj.model.w(:,i) = libsvm{i}.w;
  else
    obj.model.w(:,i) = -libsvm{i}.w;
  end
end

% apply bias multiplier if required
if obj.bias_mul ~= 1
  error('not implemented');
end

end

