classdef GenericCodebkGen < handle
  %GENERICCODEBKGEN Generic interface for training codebooks
  
  properties
  end
  
  methods(Abstract)
    [codebook,samples,mdata] = train(obj, imlist, samples, mdata)
  end
  
end

