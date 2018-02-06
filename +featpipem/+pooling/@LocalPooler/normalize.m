function [pcode] = normalize(obj, pcode)
%NORMALIZE Summary of this function goes here
%   Detailed explanation goes here

% now normalize all sub-bins
if strcmp(obj.subbin_norm_type, 'l2')
  
  pcode_norm = sqrt(sum(pcode .^ 2, 1));
  pcode_norm = max(pcode_norm, eps);
  pcode = bsxfun(@times, pcode, 1 ./ pcode_norm);
  
elseif strcmp(obj.subbin_norm_type, 'l1')
  
  pcode_norm = sum(pcode, 1);
  pcode_norm = max(pcode_norm, eps);
  pcode = bsxfun(@times, pcode, 1 ./ pcode_norm);
  
end

% vectorise
pcode = pcode(:);

% now normalize whole code
if strcmp(obj.norm_type,'l2')
  pcode = pcode/max(norm(pcode,2), eps);
elseif strcmp(obj.norm_type,'l1')
  pcode = pcode/max(norm(pcode,1), eps);
end

% now apply kernel map if specified
if obj.kermap_i ~= featpipem.Constant.SVMKER_NONE
  % (note: when adding extra kernel maps, note that the getDim function
  % must also be modified to reflect the appropriate increase in code
  % dimensionality)
  if obj.kermap_i == featpipem.Constant.SVMKER_KCHI2
    % chi-squared approximation
    pcode = vl_homkermap(pcode, 1, 'kchi2');
  elseif obj.kermap_i == featpipem.Constant.SVMKER_HELLINGER
    % "generalised" (signed) Hellinger kernel
    pcode = sign(pcode) .* sqrt(abs(pcode));
  end
  
  % now post-normalize whole code
  if strcmp(obj.post_norm_type,'l2')
    pcode = pcode/max(norm(pcode,2), eps);
  elseif strcmp(obj.post_norm_type,'l1')
    pcode = pcode/max(norm(pcode,1, eps));
  end
end

end

