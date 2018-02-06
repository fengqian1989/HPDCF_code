classdef TSRankPooler < handle & featpipem.pooling.LocalPooler
  %LOCALPOOLER Multiple local pooling
  
  properties
    quad_divs    % bin divisions
    horiz_divs
    scale_fuse
    rp_CVAL
  end
  
  properties(SetAccess = protected)
  end
  
  methods
    function obj = TSRankPooler(featextr, encoder)
      obj.subbin_norm_type = 'none';
      obj.norm_type = 'none';
      obj.post_norm_type = 'l2';
      
      obj.set_pool_type('sum');
      obj.quad_divs = [1;1];
      obj.horiz_divs = 0;
      
      obj.scale_fuse = 'sum';                    
      obj.set_kermap('hellinger');
           
      obj.rp_CVAL = 1;
      
      obj.featextr_ = featextr;
      obj.encoder_ = encoder;
    end
    
    function label = get_label(obj) %#ok<MANU>
      label = 'ltspm'; 
    end
    
    function dim = get_output_dim(obj)
      dim = sum(obj.get_block_count()*cell2mat(obj.encoder_.get_output_dim()))*2;
      % account for expansion in dimensionality when using kernel map
      if strcmp(obj.kermap,'homker')
        dim = dim*3;
      end
    end
    
    function count = get_block_count(obj)
       count = sum(obj.quad_divs(1,:) .* obj.quad_divs(2,:)); 
    end
    
    pcode = compute(obj, im)
    pcode = compute_impl(obj, imsize, assign, poolpos, locs)
    pcode = compute_rp(obj, pcode, CVAL)
    [pcodes,feat_pos] = compute_subwin(obj, feats, locs, subwin_radius)
  end
  
end

