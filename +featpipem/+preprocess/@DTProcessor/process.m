function output = process(obj, im_path)
%PROCESS Summary of this function goes here
%   Detailed explanation goes here

V = VideoReader(im_path);
vid_size = [V.Height; V.Width; V.NumberOfFrames];

[pathstr,name,ext] = fileparts(im_path);% the video path
filesepids = strfind(pathstr,filesep);
pathstr = [pathstr(1:filesepids(end)-1) '_dtf' pathstr(filesepids(end):end)];
output_path = fullfile(pathstr,[name '.bin']);% the output path
params = [obj.quit_mode, obj.track_length, obj.init_gap,...
          obj.patch_size, obj.nxy_cell, obj.nt_cell,...
          obj.start_frame, obj.end_frame, obj.min_distance,...
          obj.scale_num];

if ~exist(output_path,'file')
    fprintf('mkdir: %s',pathstr);
    mkdir(pathstr);
    outputflag = mexDenseTrackStab(im_path,output_path,params);
    assert(outputflag==0, 'Dense Trajectory failed.');
else
    bin_info = dir(output_path);
    if bin_info.bytes == 0
        delete(output_path);
        outputflag = mexDenseTrackStab(im_path,output_path,params);
        assert(outputflag==0, 'improve Dense Trajectory failed.');
    end
    fprintf('%s has existed.\n',output_path);
end

output.path = output_path;
output.vid_size = vid_size;
% obj.fea_num = length(cell_data);
end

