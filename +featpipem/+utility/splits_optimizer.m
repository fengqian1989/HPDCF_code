function splits_rank = splits_optimizer(Acc_tensor,Idx_tensor,Idxname)
% Exemple:
% splits_rank = featpipem.utility.splits_optimizer(Acc_tensor,Idx_tensor);

%% 可以未知张量阶数，保证最后一维是splits
tensor_ndim = ndims(Idx_tensor);% 张量的是几维的
split_len = size(Idx_tensor,tensor_ndim);% splits个数
Idx_tensor_cell = num2cell(Idx_tensor,[1:tensor_ndim-1]);
% 按最后一维度将张量分成cell，这样可以张量前面维度可以是任意的
for nbin = 1:split_len
    %% compute nbin histogram 
    nbin_tensor = cell2mat(Idx_tensor_cell(1:nbin));
%     nbin_tensor = Idx_tensor(:,:,:,1:nbin);
    nbin_hist = histcounts(nbin_tensor,0.5:split_len+0.5);
    [nbest_splits_count,nbest_splits_idx] = sort(nbin_hist,'descend');
%     fq_histogram(nbin_hist);
    splits_rank(nbin).split_hist = nbin_hist; % 直方图
    
    %% the former nbin accuacy tensor
    nbin_Acc_tensor = zeros(size(Acc_tensor));
    for ii = 1:nbin
        idx = find(Idx_tensor == nbest_splits_idx(ii));
        nbin_Acc_tensor(idx) = Acc_tensor(idx);
    end
    % 所有参数组合分别对应的1：nsplit的平均准确率矩阵，是个三阶张量，不显示
    nbin_Meanacc_tensor = sum(nbin_Acc_tensor,4)./nbin;
    %% best parameter combination
    % 最佳参数组合的1：nsplit平均准确率(最大值)，及其改组参数的索引位置
    [nbin_maxacc,nbin_maxacc_idx] = max(nbin_Meanacc_tensor(:)); 
    % 上述参数索引位置
    [sub_vec{1:ndims(nbin_Meanacc_tensor)}] = ...
        ind2sub(size(nbin_Meanacc_tensor),nbin_maxacc_idx);
%     sub_vec = ind2subv(size(nbin_Meanacc_tensor),nbin_maxacc_idx);
%     [sub_i,sub_j,sub_k] = ind2sub(size(nbin_Meanacc_tensor),nbin_maxacc_idx)
    % 上述平均准确率最大值细节,默认看不到该值
    nbin_maxaccvector = permute(nbin_Acc_tensor(sub_vec{:},:),[1,4,2,3]);
%     nbin_maxaccvector = permute(nbin_Acc_tensor(sub_i,sub_j,sub_k,:),[1,4,2,3]);
    %% parameter regulation
    % 
    
    %% results
    % 最佳的1：nsplit分组号
    splits_rank(nbin).nbin_Acc_tensor = nbin_Acc_tensor;
    splits_rank(nbin).nbin_idx = nbest_splits_idx(1:nbin);
    % 所有参数组合的1：nsplit的平均准确率和方差，是上面那个的均值和方差，可以表示平均水平
    splits_rank(nbin).nbin_meanacc = mean(nbin_Meanacc_tensor(:));
    splits_rank(nbin).nbin_accstd = std(nbin_Meanacc_tensor(:));% 前n位置的准确率方差
    % 最佳参数组合的1：nsplit平均准确率(最大值)、索引位置和平均准确率处细节
    splits_rank(nbin).nbin_bestacc = nbin_maxacc; 
    splits_rank(nbin).nbin_bestaccparams = sub_vec; % 参数
    splits_rank(nbin).nbin_bestaccvector = nbin_maxaccvector(nbin_maxaccvector~=0);%最大平均准确率处细节
    
    %% the valid splits 
    % 非零位置中最后一个即为有效值得位置
    valid_len = find(nbest_splits_count>0);
    valid_len = valid_len(end);
    splits_meanacc = zeros(1,valid_len);
    splits_bestacc = zeros(1,valid_len);
    for ii = 1:valid_len
        idx = find(Idx_tensor == nbest_splits_idx(ii));
        splits_meanacc(ii) = mean(Acc_tensor(idx));
        splits_bestacc(ii) = max(Acc_tensor(idx));
    end 
    
    splits_rank(nbin).splits_idx = nbest_splits_idx(1:valid_len); % 有效索引序号(split序号)
    splits_rank(nbin).splits_meanacc = splits_meanacc; % 有效索引的准确率(平均)
    splits_rank(nbin).splits_bestacc = splits_bestacc; % 有效索引的准确率(最高)
    
    %% display
    fprintf('\n =========================%d==========================',nbin)
    fprintf('\n 最佳的1：nsplit分组 :')
    fprintf('%d  ',splits_rank(nbin).nbin_idx);
    fprintf('\n 所有参数组合的1：nsplit的平均准确率和方差 :')
    fprintf('%.4f±%.4f',splits_rank(nbin).nbin_meanacc,...
        splits_rank(nbin).nbin_accstd);
    fprintf('\n ☆最佳参数组合的1：nsplit平均准确率(最大值) :')
    fprintf('%.4f  ',splits_rank(nbin).nbin_bestacc);
    fprintf('\n ☆最佳参数组合 :')
    fprintf('%d(1/%d %d) ',sub_vec{1},2^sub_vec{1},Idxname.dim1{sub_vec{1}});
    fprintf('%d(%d) ',sub_vec{2},Idxname.dim2(sub_vec{2}));
    fprintf('%d ',sub_vec{3});
    fprintf('\n ☆最佳准确率细节 :')
    fprintf('%.4f ',splits_rank(nbin).nbin_bestaccvector);
    
    [order_flag,bestacc_falg] = check_nbin(splits_rank,nbin);
    fprintf('\n\n valid index  :')
    fprintf('  %2.3f',nbest_splits_idx(1:valid_len));
    fprintf('\n mean accuracy:')
    fprintf('  %.4f',splits_meanacc);
    fprintf('\n high accuracy:')
    fprintf('  %.4f',splits_bestacc);
    fprintf('\n accuracy flag:')
    fprintf('  %.4f',order_flag);
    fprintf('\n bestaccuracy flag: %d \n',bestacc_falg);
end


function [order_flag,bestacc_falg] = check_nbin(splits_rank,nbin)

%% if the valid splits have the correct order
len = length(splits_rank(nbin).splits_meanacc);
correct_idx = 1:len;
[~,mean_idx] = sort(splits_rank(nbin).splits_meanacc,'descend');
[~,high_idx] = sort(splits_rank(nbin).splits_bestacc,'descend');
order_flag = (correct_idx - mean_idx) & (correct_idx - high_idx);

%% if the best split accuracy in the nbin best accuracy vector
split_bestacc = max(splits_rank(nbin).splits_bestacc);
nbin_bestvector = splits_rank(nbin).nbin_bestaccvector;
bestacc_falg = ismember(split_bestacc,nbin_bestvector);

