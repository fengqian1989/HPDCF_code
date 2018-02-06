function pcode = compute_impl(obj, imsize, assign, poolpos, locs)
%COMPUTE Pool features using the spatial pyramid and tempoal pyramid match kernel

%% Check the input data
if imsize(1)==0 && imsize(2)==0 && imsize(3)==0
   error('Input is not video format.'); 
end
if isempty(obj.horiz_divs)
   error('Temporal Pooling is Empty!'); 
end
if isempty(obj.quad_divs)
    error('Spatial Pooling is Empty!'); 
end
%% Tmeporal Pool Assignment and Spatial Pool Encoding
if obj.horiz_divs == 0 % Not use Temporal Pyramid
    obj.horiz_divs = [fix(imsize(3)/2);fix(imsize(3)/2)];
end
tp.level_count = size(obj.horiz_divs ,2); % Temporal Pooler only has one dimension
tp.lbin_counts = obj.horiz_divs (1,:); % bin
tp.lwin_counts = obj.horiz_divs (2,:); % windows
tp.bin_count = sum(tp.lbin_counts);
% tp.encode_len = obj.encoder_.get_output_dim();

sp.level_count = size(obj.quad_divs, 2); 
sp.lbin_counts = obj.quad_divs(1,:) .* obj.quad_divs(2,:);
sp.bin_count = sum(sp.lbin_counts);
sp.encode_len = obj.encoder_.get_output_dim();

stpcode = cellfun(@(x,y,z) tpm_pooler(obj,x,y,z,locs,imsize,tp,sp),...
    assign, poolpos, sp.encode_len, 'UniformOutput', false);

%% 外层cell是不同词典，内层cell是不同时间片段，两者交换
stpcode = cat(1,stpcode{:}); % 将嵌套cell展开，不同样本的不同词典特征排列
stpcode = num2cell(stpcode,1); % 将上述矩阵，每一类词典特征所对应的所有样本，组成一个cell

%% 将不同词典串起来
for t_num = 1:length(stpcode) % 时间片段 k

    stpcode{t_num} = cellfun(@(x) obj.encoder_.post_process(x),stpcode{t_num},...
        'UniformOutput', false);
    stpcode{t_num} = cellfun(@(x) obj.normalize(x),stpcode{t_num},'UniformOutput', false);
    
    % Apply the feature w.r.t. different dictionary
    stpcode{t_num} = cat(1,stpcode{t_num}{:});
end

pcode = cell2mat(stpcode);

end

%% TemportalPool 
%% 这里编的不简练，不同词典特征，不同维对应的pooling规则应该是相同的，
%% 所以这里feats_sel应该只算一次，不同的词典、尺度特征
function stp_code = tpm_pooler(obj, assign, poolpos, encode_len,...
    locs, imsize, tp, sp)  
% Parameters
level_count = tp.level_count;
lbin_counts = tp.lbin_counts;
lwin_counts = tp.lwin_counts;
bin_count = tp.bin_count;
% % 这里的 assign,poolpos 仍然是cell, encode_len是定值
% tp_code = repmat({zeros(encode_len, bin_count,'single')},...
%     size(poolpos));
stp_code = cell(1,bin_count);

code_idx = 0;
for ilevel = 1:level_count
    lwin_count = lwin_counts(ilevel);
    tunit = 1 / lwin_count;
    
    lbin_count = lbin_counts(ilevel);
    binids = mod(floor(locs(3,:)/ tunit), lbin_count) +1;
    for ianchor = 1:lbin_count
        code_idx = code_idx + 1;
        feats_sel = find(binids == ianchor);
       %% 这里添加spatial Pooling 这里在cellfun计算不同词典、尺度特征
       %% cellfun处理词典那层cell，尺度cell在mulscalefuse函数中处理
        stp_code{code_idx} = cellfun(@(x,y) spm_pooler(obj,x(:,feats_sel),...
            y(:,feats_sel),encode_len,locs(:,feats_sel),imsize,sp),assign,...
            poolpos,'UniformOutput', false);
    end
end

%% Fuse multiple scale features
stp_code = cellfun(@(x) mulscalefuse(obj,x),stp_code,'UniformOutput', false);

end

%% SPMPool 
function sp_code = spm_pooler(obj,assign,poolpos,encode_len,...
    locs,imsize,sp)
% Parameters
level_count = sp.level_count; 
lbin_counts = sp.lbin_counts;
bin_count = sp.bin_count;

sp_code = zeros(encode_len, bin_count,'single');

code_idx = 0;
for ilevel = 1:level_count
    bin_quad_divs = obj.quad_divs(:,ilevel);
    lbin_count = lbin_counts(ilevel);
    wunit = 1 / bin_quad_divs(2);
    hunit = 1 / bin_quad_divs(1);
    xbin = ceil(locs(1,:) / wunit);
    ybin = ceil(locs(2,:) / hunit);
    binids = (xbin - 1) * bin_quad_divs(2) + ybin;
    for ianchor = 1:lbin_count
        code_idx = code_idx + 1;
        feats_sel = (binids == ianchor);
        if any(feats_sel ~= 0)
            sp_code(:,code_idx) = featpipem.lib.mexPooler( assign,...
                int32(poolpos), encode_len, obj.pool_type_i, feats_sel );
        else
            warning('SPMPool:EmptyBin','empty bin!');
        end
    end
end

end

%% strategy for multiple scales feature
function pcode = mulscalefuse(obj,spcode)

% Apply fusion towards multiple scale pooling vector
switch obj.scale_fuse
    case 'sum'
        spcode_sum = spcode{1};
        for i = 2:length(spcode)
            spcode_sum = spcode_sum + spcode{i};
        end
        pcode = spcode_sum;
    otherwise
        % empty operation
end

end


