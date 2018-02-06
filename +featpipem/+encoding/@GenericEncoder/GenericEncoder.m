classdef GenericEncoder < handle
  %GENERICENCODER Generic interface to bag-of-words encoder
  
  properties
  end
  
  methods(Abstract)
    dim = get_input_dim(obj)
    dim = get_output_dim(obj)
    [assign,poolpos] = encode(obj, feats)
  end
  
  methods
    function pcode = post_process(obj, pcode) % #ok<MANU>
    end
  end
  
end

