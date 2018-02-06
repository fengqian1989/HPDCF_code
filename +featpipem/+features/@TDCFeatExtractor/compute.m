function [feats, frames] = compute(obj, im)
%COMPUTE Summary of this function goes here
%   Detailed explanation goes here

% frames: [location x1 ...; location y1 ...; temporal t1 ...; 
%          displament x1 ...; displament y1 ...]

%% input and output paths
% bin file for improve dense trajectory------------------------------------
file_bin = im;
fid = fopen(file_bin,'rb');
feat = fread(fid,[obj.dtf_lens('info') + obj.tra_point_len + obj.tra_disp_len + ...
    obj.dtf_lens('hog') + obj.dtf_lens('hof') + obj.dtf_lens('mbhx') + ...
    obj.dtf_lens('mbhy'), inf],'float');
fclose(fid);
[im_path,pathstr_tdc,tdc_output_path] = gentdcpath(file_bin);
%%  multi-scale convolutional maps
cmaps = obj.mul_conv_maps;
use_gpu = obj.use_gpu;
rgb_tag = obj.features('rgb');
flow_tag = obj.features('flow');
idt_tag = obj.features('idt');
%% if the tdc file had existed
if exist(tdc_output_path,'file')
    tdc_file_info = dir(tdc_output_path);
    if tdc_file_info.bytes == 0
        delete(tdc_output_path);
    else
        fprintf(['skip the tdc feature extracting.\n'...
            'tdc file %s has existed.\n'],tdc_output_path);
        load(tdc_output_path);
    end
else
%% or compute the TDC and iDT features
    fprintf('mkdir: %s.\n',pathstr_tdc);
    mkdir(pathstr_tdc);
    if isempty(feat)
        %% DT preprocessor
        preproc = obj.preproc; % DT parameters
        params = [preproc.quit_mode, preproc.track_length, preproc.init_gap,...
            preproc.patch_size, preproc.nxy_cell, preproc.nt_cell,...
            preproc.start_frame, preproc.end_frame, preproc.min_distance,...
            preproc.scale_num];
        outputflag = mexDenseTrackStab(im_path,file_bin,params); % DT compute
        assert(outputflag==0, 'Dense Trajectory failed.');
        %% idt feature
        fid = fopen(file_bin,'rb');
        feat = fread(fid,[obj.dtf_lens('info') + obj.tra_point_len + ...
            obj.tra_disp_len + obj.dtf_lens('hog') + obj.dtf_lens('hof') +...
            obj.dtf_lens('mbhx') + obj.dtf_lens('mbhy'), inf],'float');
        fclose(fid);
        if isempty(feat)
            error('%s cannot extract DT feature.\n',file_bin);
        end
    end
    %% Hand-craft improve dense trajectory
    idt.info = feat(1:obj.dtf_lens('info'),:);% contains 10 values:
    % 1 frame_num| 2 mean_location_x| 3 mean_location_y|
    % 4 var_location_x| 5 var_location_y| 6 length| 7 cscale
    % 8 norm_mean_location_x| 9 norm_mean_location_y| 10 norm_location_t
    idt.tra_point = feat(obj.dtf_lens('info')+1:obj.dtf_lens('info')+obj.tra_point_len,:);
    % trajectory location x y arranged by frame
    idt.tra_disp = feat(obj.dtf_lens('info')+obj.tra_point_len+1 :...
        obj.dtf_lens('info')+obj.tra_point_len+obj.tra_disp_len,:);
    % trajectory displacement x y arranged by frame
    feat_start = obj.feat_start;
    idt.hog = single(feat(feat_start+1 :feat_start+obj.dtf_lens('hog'),:));
    feat_start = feat_start+obj.dtf_lens('hog');
    idt.hof = single(feat(feat_start+1 :feat_start+obj.dtf_lens('hof'),:));
    feat_start = feat_start+obj.dtf_lens('hof');
    idt.mbhx = single(feat(feat_start+1:feat_start+obj.dtf_lens('mbhx'),:));
    feat_start = feat_start+obj.dtf_lens('mbhx');
    idt.mbhy = single(feat(feat_start+1:feat_start+obj.dtf_lens('mbhy'),:));
    % whole feature vector
    idt.fea = single(feat(obj.feat_start+1:end,:));
    
    %% DenseFlow
    [xFlowImgs,yFlowImgs,orgImgs] = mexDenseFlow(im_path,obj.df_bound,obj.df_quiet_mode);
    %% Spatial TDC Feature   
    if ~isempty(rgb_tag)
        display('Extract spatial TDC Feature...');
        %rgb_tag = {'pool1','pool2','conv3','conv4','conv5'};
        model_file = './lib/TDD_net/model_pretrain/spatial_v2.caffemodel';
        sp_convs = {};
        for sn = 1:size(cmaps,1)
            model_def_file = [ './lib/TDD_net/model_proto/spatial_net_scale_', num2str(sn), '.prototxt'];
            caffe.reset_all();
            if exist('use_gpu', 'var') && use_gpu
                caffe.set_mode_gpu();
                gpu_id = 0;  % we will use the first gpu in this demo
                caffe.set_device(gpu_id);
            else
                caffe.set_mode_cpu();
            end
            net = caffe.Net(model_def_file, model_file, 'test');
            
            sp_convs = SpatialCNNFeature(im_path, net, cmaps(sn,3), cmaps(sn,4), rgb_tag);
            if max(idt.info(1,:)) > size(sp_convs{1},4)
                ind =  idt.info(1,:) <= size(sp_convs{1},4);
                idt.info = idt.info(:,ind);
                idt.tra_point = idt.tra_point(:,ind);
            end
            
            [sp_convs_norm_st, sp_convs_norm_ch] = cellfun(@(x) FeatureMapNormalization(x),...
                sp_convs, 'UniformOutput', false);
            tdc_sp_convs_norm_st = cellfun(@(x) TDD(idt.info, idt.tra_point,...
                x,cmaps(sn,1),cmaps(sn,2),1),sp_convs_norm_st,'UniformOutput', false);
            tdc_sp_convs_norm_ch = cellfun(@(x) TDD(idt.info, idt.tra_point,...
                x,cmaps(sn,1),cmaps(sn,2),1),sp_convs_norm_ch,'UniformOutput', false);
            
            for nlayer = 1:length(rgb_tag)
                tdc.(['sp_' rgb_tag{nlayer} 'nst']){sn} =  single(tdc_sp_convs_norm_st{nlayer});
                tdc.(['sp_' rgb_tag{nlayer} 'nch']){sn} =  single(tdc_sp_convs_norm_ch{nlayer});
            end
        end
    end
    clear sp_convs sp_convs_norm_st sp_convs_norm_ch tdc_sp_convs_norm_st tdc_tp_convs_norm_ch;
    %% Temporal TDC Feature
    if ~isempty(flow_tag)
%         flow_tag = {'pool1','pool2','conv3','conv4','conv5'};
        display('Extract temporal TDC Feature...');
        model_file = './lib/TDD_net/model_pretrain/temporal_v2.caffemodel';
        tp_convs = {};
        for sn = 1:size(cmaps,1)
            model_def_file = [ './lib/TDD_net/model_proto/temporal_net_scale_', num2str(sn),'.prototxt'];
            caffe.reset_all();
            if exist('use_gpu', 'var') && use_gpu
                caffe.set_mode_gpu();
                gpu_id = 0;  % we will use the first gpu in this demo
                caffe.set_device(gpu_id);
            else
                caffe.set_mode_cpu();
            end
            net = caffe.Net(model_def_file, model_file, 'test');
            
            tp_convs = TemporalCNNFeature(xFlowImgs,yFlowImgs,net,cmaps(sn,3),cmaps(sn,4),flow_tag);
            if max(idt.info(1,:)) > size(tp_convs{1},4)
                ind =  idt.info(1,:) <= size(tp_convs{1},4);
                idt.info = idt.info(:,ind);
                idt.tra_point = idt.tra_point(:,ind);
            end
            
            [tp_convs_norm_st, tp_convs_norm_ch] = cellfun(@(x) FeatureMapNormalization(x),...
                tp_convs, 'UniformOutput', false);
            tdc_tp_convs_norm_st = cellfun(@(x) TDD(idt.info, idt.tra_point,...
                x,cmaps(sn,1),cmaps(sn,2),1),tp_convs_norm_st,'UniformOutput', false);
            tdc_tp_convs_norm_ch = cellfun(@(x) TDD(idt.info, idt.tra_point,...
                x,cmaps(sn,1),cmaps(sn,2),1),tp_convs_norm_ch,'UniformOutput', false);
            
            for nlayer = 1:length(flow_tag)
                tdc.(['tp_' flow_tag{nlayer} 'nst']){sn} =  single(tdc_tp_convs_norm_st{nlayer});
                tdc.(['tp_' flow_tag{nlayer} 'nch']){sn} =  single(tdc_tp_convs_norm_ch{nlayer});
            end
        end
    end
    clear tp_convs tp_convs_norm_st tp_convs_norm_ch tdc_tp_convs_norm_st tdc_tp_convs_norm_ch;
    %% Frame information 
    idx = idt.info(7,:)==1;
    frames = [idt.info(8:10,idx)];% update
    save(tdc_output_path,'idt','tdc','frames','-v7.3');
end

feats = {};
rgb_tag = obj.features('rgb');
flow_tag = obj.features('flow');
idt_tag = obj.features('idt');
for nlayer = 1:length(rgb_tag)
    feats = [feats; {tdc.(['sp_' rgb_tag{nlayer} 'nst'])}; ...
        {tdc.(['sp_' rgb_tag{nlayer} 'nch'])}];
end
for nlayer = 1:length(flow_tag)
    feats = [feats; {tdc.(['tp_' flow_tag{nlayer} 'nst'])}; ...
        {tdc.(['tp_' flow_tag{nlayer} 'nch'])}];
end
idt_feat = [];
for nfeat = 1:length(idt_tag)
    idt_feat = single([idt_feat;idt.(idt_tag{nfeat});]);
end
if ~isempty(idt_feat)
    idt_feat = {idt_feat};
    feats = [feats; {idt_feat};];% 加了一层cell
end

%% 降维 这里是在编码时候，用于单个图像或视频，还需要改
if obj.low_proj_enabled && ~isempty(obj.low_proj)
  % dimensionality reduction
  feats = cellfun(@(x,y) mulscaledimred(x,y), obj.low_proj,feats,...
      'UniformOutput', false);
end

feats = obj.post_process(feats);

end

%% 这里是编码时候使用，此时不同尺度的特征存储于不同cell中，
%% PCA project in encoding phase
function dimred_feat = mulscaledimred(low_proj,mulfeats)

dimred_feat = cellfun(@(x) low_proj*x, mulfeats,'UniformOutput', false);

end

%% generate the video path and tdc path
function [im_path,pathstr_tdc,tdc_output_path] = gentdcpath(file_bin)

% avi file for video data--------------------------------------------------
[pathstr,name,ext] = fileparts(file_bin);% the bin path
filesepids = strfind(pathstr,filesep); % separator locations
dft_path = pathstr(filesepids(end-1)+1:filesepids(end));%
avi_path = dft_path(1:end-5);
pathstr_avi = [pathstr(1:filesepids(end-1)) avi_path pathstr(filesepids(end):end)];
im_path = fullfile(pathstr_avi,[name '.avi']);% the avi path
% mat file for TDC feature-------------------------------------------------
tdc_path = [avi_path '_tdc'];
pathstr_tdc = [pathstr(1:filesepids(end-1)) tdc_path pathstr(filesepids(end):end)];
name_tdc = sprintf('%s.mat',name);
tdc_output_path = fullfile(pathstr_tdc,name_tdc);% the output path

end
