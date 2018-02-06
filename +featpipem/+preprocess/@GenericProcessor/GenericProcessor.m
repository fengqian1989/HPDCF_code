classdef GenericProcessor < handle
  %GENERICPROCESSOR Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods(Abstract)
    im = process(obj, im)
  end
  
end

