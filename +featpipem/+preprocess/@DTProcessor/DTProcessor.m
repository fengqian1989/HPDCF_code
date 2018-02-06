classdef DTProcessor < handle & featpipem.preprocess.GenericProcessor
  %NORMALPROCESSOR Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    fea_num
    % tracking parameter
    track_length
    init_gap
    % feature parameter
    patch_size
    nxy_cell
    nt_cell
    % pyramid parameter
    scale_num
    % frame parameter
	start_frame
	end_frame
	min_distance
    % calibration parameter
    traj_len
    % output
	quit_mode
  end
  
  methods
    function obj = DTProcessor()
        
       obj.fea_num = 0;
       % tracking parameter
       obj.track_length = 15;
       obj.init_gap = 1;
       % feature parameter
       obj.patch_size = 32;
       obj.nxy_cell = 2;
       obj.nt_cell = 3;
       % pyramid parameter
       obj.scale_num = 8;
       % frame parameter
	   obj.start_frame = 0;
	   obj.end_frame = 1000000;
	   obj.min_distance = 5;
       % calibration parameter
       obj.traj_len = 10+4*obj.track_length+3*96 +108;
       % output
	   obj.quit_mode = 1;  
    end
    
    output = process(obj, im)
  end
  
end

