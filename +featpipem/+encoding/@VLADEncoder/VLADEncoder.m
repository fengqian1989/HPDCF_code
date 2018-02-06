classdef VLADEncoder < handle & featpipem.encoding.GenericEncoder
  %FVENCODER Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    max_comps % -1 for exact
    num_nn    % number of nearest neighbour bases to assign to
    power_norm 
    norm_type % Global or Intra-normalization
    codebook_
  end
  properties(SetAccess = protected)
    kdtree_
  end
  
  methods
    function obj = VLADEncoder(codebook)      
      % set default parameter values
      obj.max_comps = 500;
      obj.num_nn = 1;%5;
      obj.power_norm = 0.5;
      obj.norm_type = 'Global';% 'Intra' or 'Global';
      
      % setup encoder
      obj.codebook_ = codebook;
%       obj.kdtree_ = vl_kdtreebuild(obj.codebook_);
      obj.kdtree_ = cellfun(@(x) vl_kdtreebuild(x),codebook,...
          'UniformOutput',false);
    end
    
    function label = get_label(obj) %#ok<MANU>
      label = 'vlad';
    end
    
    function dim = get_input_dim(obj)
      dim = cellfun(@(x) size(x,1),obj.codebook_,'UniformOutput',false);
    end
    
    function dim = get_codebook_dim(obj)
      dim = cellfun(@(x) size(x,2),obj.codebook_,'UniformOutput',false);
    end
    
    function dim = get_output_dim(obj)
%       dim = obj.get_codebook_dim() * get_input_dim(obj);
      dim = cellfun(@(x,y) x*y, obj.get_codebook_dim(),...
          obj.get_input_dim(),'UniformOutput',false);
    end
    
    % compute encoding
    [assign,poolpos] = encode(obj, feats)
    
    pcode = post_process(obj, pcode)
  end
  
end

