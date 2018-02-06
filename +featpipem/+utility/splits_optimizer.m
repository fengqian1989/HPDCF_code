function splits_rank = splits_optimizer(Acc_tensor,Idx_tensor,Idxname)
% Exemple:
% splits_rank = featpipem.utility.splits_optimizer(Acc_tensor,Idx_tensor);

%% ����δ֪������������֤���һά��splits
tensor_ndim = ndims(Idx_tensor);% �������Ǽ�ά��
split_len = size(Idx_tensor,tensor_ndim);% splits����
Idx_tensor_cell = num2cell(Idx_tensor,[1:tensor_ndim-1]);
% �����һά�Ƚ������ֳ�cell��������������ǰ��ά�ȿ����������
for nbin = 1:split_len
    %% compute nbin histogram 
    nbin_tensor = cell2mat(Idx_tensor_cell(1:nbin));
%     nbin_tensor = Idx_tensor(:,:,:,1:nbin);
    nbin_hist = histcounts(nbin_tensor,0.5:split_len+0.5);
    [nbest_splits_count,nbest_splits_idx] = sort(nbin_hist,'descend');
%     fq_histogram(nbin_hist);
    splits_rank(nbin).split_hist = nbin_hist; % ֱ��ͼ
    
    %% the former nbin accuacy tensor
    nbin_Acc_tensor = zeros(size(Acc_tensor));
    for ii = 1:nbin
        idx = find(Idx_tensor == nbest_splits_idx(ii));
        nbin_Acc_tensor(idx) = Acc_tensor(idx);
    end
    % ���в�����Ϸֱ��Ӧ��1��nsplit��ƽ��׼ȷ�ʾ����Ǹ���������������ʾ
    nbin_Meanacc_tensor = sum(nbin_Acc_tensor,4)./nbin;
    %% best parameter combination
    % ��Ѳ�����ϵ�1��nsplitƽ��׼ȷ��(���ֵ)������������������λ��
    [nbin_maxacc,nbin_maxacc_idx] = max(nbin_Meanacc_tensor(:)); 
    % ������������λ��
    [sub_vec{1:ndims(nbin_Meanacc_tensor)}] = ...
        ind2sub(size(nbin_Meanacc_tensor),nbin_maxacc_idx);
%     sub_vec = ind2subv(size(nbin_Meanacc_tensor),nbin_maxacc_idx);
%     [sub_i,sub_j,sub_k] = ind2sub(size(nbin_Meanacc_tensor),nbin_maxacc_idx)
    % ����ƽ��׼ȷ�����ֵϸ��,Ĭ�Ͽ�������ֵ
    nbin_maxaccvector = permute(nbin_Acc_tensor(sub_vec{:},:),[1,4,2,3]);
%     nbin_maxaccvector = permute(nbin_Acc_tensor(sub_i,sub_j,sub_k,:),[1,4,2,3]);
    %% parameter regulation
    % 
    
    %% results
    % ��ѵ�1��nsplit�����
    splits_rank(nbin).nbin_Acc_tensor = nbin_Acc_tensor;
    splits_rank(nbin).nbin_idx = nbest_splits_idx(1:nbin);
    % ���в�����ϵ�1��nsplit��ƽ��׼ȷ�ʺͷ���������Ǹ��ľ�ֵ�ͷ�����Ա�ʾƽ��ˮƽ
    splits_rank(nbin).nbin_meanacc = mean(nbin_Meanacc_tensor(:));
    splits_rank(nbin).nbin_accstd = std(nbin_Meanacc_tensor(:));% ǰnλ�õ�׼ȷ�ʷ���
    % ��Ѳ�����ϵ�1��nsplitƽ��׼ȷ��(���ֵ)������λ�ú�ƽ��׼ȷ�ʴ�ϸ��
    splits_rank(nbin).nbin_bestacc = nbin_maxacc; 
    splits_rank(nbin).nbin_bestaccparams = sub_vec; % ����
    splits_rank(nbin).nbin_bestaccvector = nbin_maxaccvector(nbin_maxaccvector~=0);%���ƽ��׼ȷ�ʴ�ϸ��
    
    %% the valid splits 
    % ����λ�������һ����Ϊ��Чֵ��λ��
    valid_len = find(nbest_splits_count>0);
    valid_len = valid_len(end);
    splits_meanacc = zeros(1,valid_len);
    splits_bestacc = zeros(1,valid_len);
    for ii = 1:valid_len
        idx = find(Idx_tensor == nbest_splits_idx(ii));
        splits_meanacc(ii) = mean(Acc_tensor(idx));
        splits_bestacc(ii) = max(Acc_tensor(idx));
    end 
    
    splits_rank(nbin).splits_idx = nbest_splits_idx(1:valid_len); % ��Ч�������(split���)
    splits_rank(nbin).splits_meanacc = splits_meanacc; % ��Ч������׼ȷ��(ƽ��)
    splits_rank(nbin).splits_bestacc = splits_bestacc; % ��Ч������׼ȷ��(���)
    
    %% display
    fprintf('\n =========================%d==========================',nbin)
    fprintf('\n ��ѵ�1��nsplit���� :')
    fprintf('%d  ',splits_rank(nbin).nbin_idx);
    fprintf('\n ���в�����ϵ�1��nsplit��ƽ��׼ȷ�ʺͷ��� :')
    fprintf('%.4f��%.4f',splits_rank(nbin).nbin_meanacc,...
        splits_rank(nbin).nbin_accstd);
    fprintf('\n ����Ѳ�����ϵ�1��nsplitƽ��׼ȷ��(���ֵ) :')
    fprintf('%.4f  ',splits_rank(nbin).nbin_bestacc);
    fprintf('\n ����Ѳ������ :')
    fprintf('%d(1/%d %d) ',sub_vec{1},2^sub_vec{1},Idxname.dim1{sub_vec{1}});
    fprintf('%d(%d) ',sub_vec{2},Idxname.dim2(sub_vec{2}));
    fprintf('%d ',sub_vec{3});
    fprintf('\n �����׼ȷ��ϸ�� :')
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

