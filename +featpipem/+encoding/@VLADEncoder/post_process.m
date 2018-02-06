function pcode = post_process(obj, pcode)
%POST_PROCESS Summary of this function goes here
%   Detailed explanation goes here

% Normlization according to the norm_type
switch obj.norm_type
   
    case 'Global'
        pcode = sign(pcode) .* ( abs(pcode) .^ obj.power_norm );
        
    case 'Intra'
        featureDim = obj.get_input_dim();
        codebookSize = obj.get_codebook_dim();
        spm_divs = size(pcode, 2);
        
        cellpcode = {};
        cellpcode = mat2cell(pcode, repmat(featureDim, [1, codebookSize]),...
                                ones(1, spm_divs));
        cellpcode = cellfun(@(x) x/max(norm(x,2), eps), cellpcode,...
                                'UniformOutput', false);
        
        pcode = cell2mat(cellpcode);
        
%         cellpcode = {};
%         cellpcode = mat2cell(pcode, 1, ones(1, spm_divs));                        
%         cellpcode = cellfun(@(x) sign(x).* (x./ norm(x)), cellpcode,...
%                                  'UniformOutput', false);
%         
%         pcode = [];
%         pcode = cell2mat(cellpcode);
        
        
    otherwise
        error('There is not this type normlization !!');
    
end



end

