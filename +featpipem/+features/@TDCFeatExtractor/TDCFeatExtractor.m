classdef TDCFeatExtractor < handle & featpipem.features.GenericFeatExtractor
  %PHOWEXTRACTOR Feature extractor for cell deformation
  
  properties
    use_gpu
    features = containers.Map({'flow','rgb','idt'},...
       {{'pool1','pool2','conv3','conv4','conv5'},...
       {'pool1','pool2','conv3','conv4','conv5'},...
       {'hog','hof','mbhx','mbhy'}});  
    % dense flow
    df_bound 
    df_quiet_mode
    % dense trajectory feature  
    idt_dim
    % dimensionality reducing projection
    low_proj_enabled
    low_proj 
    % name
    idt_tag
    flow_tag
    rgb_tag 
    opt_tag
    feat_dims
  end
  
  properties(SetAccess = protected)
    name = 'tdc';
    mul_conv_maps = [8,8, 480,640; 11.4286,11.4286, 340,454;...% multi-scale
    16,16, 240,320; 22.8571,24, 170,227; 32,34.2587, 120,160]; % convolutional maps
    % dense trajectory feature 
    dtf_lens = containers.Map({'info','hog','hof','mbhx','mbhy'},...
                                [10,   96,   108,    96,    96]);    
    tra_point_len
    tra_disp_len
    feat_start
    % preprocessor
    preproc %preproc_
    % dimensionality reducing projection
    layer_dims = containers.Map(...
        {'sp_pool1','sp_pool2','sp_conv3','sp_conv4','sp_conv5',...
         'tp_pool1','tp_pool2','tp_conv3','tp_conv4','tp_conv5'},...
        [    96,       256,        512,       512,       512,...
             96,       256,        512,       512,       512]);
  end
  
  methods
    function obj = TDCFeatExtractor(varargin)
      % trajectory-pooled deep-convolutional descriptors
      obj.use_gpu = 0;  % need Caffe-gpu
      % dense flow
      obj.df_bound = 15; 
      obj.df_quiet_mode = 1;
      % preprocessor
      obj.preproc = varargin{1};
      % dense trajectory feature    
      obj.tra_point_len = 2*varargin{1}.track_length;
      obj.tra_disp_len = 2*varargin{1}.track_length;
      obj.feat_start = obj.dtf_lens('info') +4*varargin{1}.track_length;
      
      obj.idt_dim = sum(cell2mat(values(obj.dtf_lens,obj.features('idt'))));
      % dimensionality reducing projection  
      obj.low_proj_enabled = 1;% dimension reduce
      obj.low_proj = [];% dimension reduce

      % GenericFeatExtractor parameters
      obj.norm_type = 'none';

      obj.opt_tag = obj.feat_opt_tag();
      obj.feat_dims = obj.get_feat_dims();
      obj.out_dim = obj.get_output_dim();
    end
    
    function obj = reset(obj)
        obj.opt_tag = obj.feat_opt_tag();
        obj.idt_dim = sum(cell2mat(values(obj.dtf_lens,obj.features('idt'))));
        obj.feat_dims = obj.get_feat_dims();
        obj.out_dim = obj.get_output_dim();
    end
    
    function opt_tag = feat_opt_tag(obj)
        rgb_tag = sprintf('%d',ismember({'pool1','pool2','conv3','conv4',...
            'conv5'},obj.features('rgb')));
        flow_tag = sprintf('%d',ismember({'pool1','pool2','conv3','conv4',...
            'conv5'},obj.features('flow')));
        opt_tag = sprintf('s%st%si%d',rgb_tag,flow_tag,~isempty(obj.features('idt')));
    end
    
    function feat_dims = get_feat_dims(obj)
        sp_dim = values(obj.layer_dims, strcat('sp_',obj.features('rgb')));% spatial feature dimension
        tp_dim = values(obj.layer_dims, strcat('tp_',obj.features('flow')));% temporal feature dimension
        idt_dim = sum(cell2mat(values(obj.dtf_lens,obj.features('idt'))));% idt feature dimension
        feat_dims = [cat(2,sp_dim{:}) cat(2,tp_dim{:}) idt_dim];
    end
    
    function dim = get_output_dim(obj) 
        feat_dims = obj.get_feat_dims();
        dim = sum(feat_dims); 
    end
        
    function low_proj = get_low_proj(obj)
      low_proj = obj.low_proj;
    end
    
    function set_low_proj(obj, low_proj)
      obj.low_proj = low_proj;
    end 
    
    [feats, frames] = compute(obj, im);
  end
  
end

