%% Add Paths
currentFolder = pwd;
setenv('PATH', [getenv('PATH'), ';', fullfile(currentFolder,'/external/caffe_cpu')]);
addpath(fullfile('./external/caffe_cpu/matcaffe'));
addpath(fullfile('./external/liblinear-2.1/windows'));
addpath(genpath(fullfile('./external/libsvm-3.18 with svdd')));
run ./external/vlfeat-0.9.16/toolbox/vl_setup
addpath(genpath(fullfile('./lib')));%子路径

%% Load Data
%(1) Conv2 %==========================================================================================
% codebook file:: codebkgen object and cookbook 
load('./data/codebook/kmeans_512_r64_f100000.00k_tdd_opt_s00000t01000i0.mat') % Conv2 in Temporal Net
% pca project matrix: low_proj
load('./data/dimred/feat_pca_tdd_opt_s00000t01000i0.mat') % for Conv2 features
% classifier 
load('./data/classifier/tdd64_opts00000t01000i0_Global_vlad512_train20_cs1_classifier.mat') % for Conv2 features

%(2) Conv3 %==========================================================================================
% load('./data/codebook/kmeans_512_r64_f100000.00k_tdd_opt_s00000t00100i0.mat') % Conv3 in Temporal Net
% load('./data/dimred/feat_pca_tdd_opt_s00000t00100i0.mat') % for Conv3 features
% load('./data/classifier/tdd64_opts00000t00100i0_Global_vlad512_train20_cs1_classifier.mat') % for Conv3 features

%(3) Conv4  %==========================================================================================
% load('./data/codebook/kmeans_512_r64_f100000.00k_tdd_opt_s00000t00010i0.mat') % Conv4 in Temporal Net
% load('./data/dimred/feat_pca_tdd_opt_s00000t00010i0.mat') % for Conv4 features
% load('./data/classifier/tdd64_opts00000t00010i0_Global_vlad512_train20_cs1_classifier.mat') % for Conv4 features

% We save the Convolutional features in ".\data\videosamples_tdc". Before using Conv3 or Conv4 features
% instand of Conv2 features, please delete the above directory

%% Dense Trajectory
% extracting Dense trajectories with default parameters
preproc = featpipem.preprocess.DTProcessor();

%% Trajectory-Pooled Deep Convolutional Feature
% In the paper, we employ temporal net feature, but this class TDCFeatExtractor
% also supports spatial net feature, or dense trajectory feature. If
% needed,please refer to the defination of this class
featextr = featpipem.features.TDCFeatExtractor(preproc);

%==========================================================================
featextr.features('flow') = {'pool2'};   %  Conv2 in Temporal Net
% featextr.features('flow') = {'conv3'}; %  Conv3 in Temporal Net
% featextr.features('flow') = {'conv4'}; %  Conv4 in Temporal Net

featextr.features('rgb') = {};
featextr.features('idt') = {};
featextr.low_proj = cellfun(@(x) x(1:32,:), low_proj,'UniformOutput',false); 
featextr.use_gpu = 0; % set to 1, if using GPU

%% VLAD Spatial Aggregation
encoder = featpipem.encoding.VLADEncoder(codebook);  

%% Rank Pooling
pooler = featpipem.pooling.TSRankPooler(featextr,encoder);% 赋值池话对象

%% classification
classifier = featpipem.classification.svm.LibLinearSvm();


