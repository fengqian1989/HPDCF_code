classdef LocalPooler < handle & featpipem.pooling.GenericPooler
  %LOCALPOOLER Multiple local pooling
  
  properties
    subbin_norm_type    % 'l1' or 'l2' (or other value = none)
    norm_type    % 'l1' or 'l2' (or other value = none)
    post_norm_type    % 'l1' or 'l2' (or other value = none)
  end
  properties(SetAccess = protected)
    pool_type    % 'sum' or 'max'
    pool_type_i
    kermap  % 'homker', 'hellinger' (or other value = none [default])
    kermap_i
    featextr_
    encoder_     % implementation of featpipem.encoding.GenericEncoder
  end
  
  methods(Abstract)
    count = get_block_count(obj)
  end
  
  methods
    function set_pool_type(obj, pool_type)
      obj.pool_type = pool_type;
      obj.pool_type_i = ...
        featpipem.Constant.parse_pool_type(pool_type);
    end
    function set_kermap(obj, kermap)
      obj.kermap = kermap;
      obj.kermap_i = ...
        featpipem.Constant.parse_svmker_type(kermap);
    end
    
    pcode = compute_block(obj, assign, poolpos, locs, blk_count)
    [pcode,params] = compute_node(obj, assign, poolpos, locs, nodes)
    pcode = compute_part(obj, assign, poolpos, locs, anchors, radius)
    
    pcode = normalize(obj, pcode)
  end
  
end

