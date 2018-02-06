close all, clear, clc, warning off

% define framework
HPDCF_setup

% input file
vid_path = fullfile('./data/videosamples/level0', 'COMP_0105_1221.avi');
% vid_path = fullfile('./data/videosamples/level0', 'COMP_0108_1016.avi');
% vid_path = fullfile('./data/videosamples/level0', 'COMP_0112_1658.avi');

output = preproc.process(vid_path);
% extract Hierarchical-Pooled Deep-Convolutional Features
pooling_vector = pooler.compute(output);
% classifier
classifier.set_model(model);

[label,~] = classifier.test(pooling_vector);

fprintf('\n\nTest sample is %s', vid_path);
fprintf(['\nPredict label is %d, corresponding level %d. \n'],label, label-1);


% vid_path = {fullfile('./data/videosamples/level0', 'COMP_0105_1221.avi');
%             fullfile('./data/videosamples/level0', 'COMP_0108_1016.avi');
% 	          fullfile('./data/videosamples/level0', 'COMP_0112_1658.avi');
%             fullfile('./data/videosamples/level0', 'COMP_0121_1416.avi');
%             fullfile('./data/videosamples/level0', 'COMP_0119_1009.avi');
% 	          fullfile('./data/videosamples/level1', 'COMP_0104_1230.avi');
% 	          fullfile('./data/videosamples/level1', 'COMP_0112_1413.avi');
%             fullfile('./data/videosamples/level1', 'COMP_0117_1150.avi');
%             fullfile('./data/videosamples/level1', 'COMP_0119_1004.avi');
%             fullfile('./data/videosamples/level1', 'COMP_0119_1130.avi');
%             fullfile('./data/videosamples/level2', 'COMP_0104_1340.avi');
%             fullfile('./data/videosamples/level2', 'COMP_0104_1456.avi');
%             fullfile('./data/videosamples/level2', 'COMP_0108_1528.avi');
%             fullfile('./data/videosamples/level2', 'COMP_0111_1251.avi');
%             fullfile('./data/videosamples/level2', 'COMP_0117_1138.avi');
%             fullfile('./data/videosamples/level3', 'COMP_0104_1557.avi');
% 	          fullfile('./data/videosamples/level3', 'COMP_0107_1314.avi');
%             fullfile('./data/videosamples/level3', 'COMP_0109_1041.avi');
%             fullfile('./data/videosamples/level3', 'COMP_0114_1554.avi');
%             fullfile('./data/videosamples/level3', 'COMP_0117_1136.avi');
%             };
% 
% for i = 1:length(vid_path)
%     tic
%     output = preproc.process(vid_path{i});
%     % extract Hierarchical-Pooled Deep-Convolutional Features
%     pooling_vector = pooler.compute(output);
%     % classifier
%     classifier.set_model(model);
%     
%     [label,~] = classifier.test(pooling_vector);
%     
%     fprintf('\nTest sample is %s', vid_path{i});
%     fprintf(['\nPredict label is %d, corresponding to level %d. \n\n'],label, label-1);
%     toc
% end






