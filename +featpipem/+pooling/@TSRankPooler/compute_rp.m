function pcode = compute_rp(obj, pcode, CVAL)
%COMPUTE Rank Pool feature
% Inputs → data   : row vector data matrix of the sequence
%           CVAL   : C value [set to 1]
% Output → pcode  : Sequence representation
% ----------
if nargin < 3
    CVAL = obj.rp_CVAL;
end
% ----------
data = pcode';% 按行排列
%% 
OneToN = [1:size(data,1)]';
Data = cumsum(data);
Data = Data ./ repmat(OneToN,1,size(Data,2));
W_fow = liblinearsvr(getNonLinearity(Data),CVAL,2); clear Data;
order = 1:size(data,1);
[~,order] = sort(order,'descend');
data = data(order,:);
Data = cumsum(data);
Data = Data ./ repmat(OneToN,1,size(Data,2));
W_rev = liblinearsvr(getNonLinearity(Data),CVAL,2);
pcode = [W_fow ; W_rev]; %还原成一列

end


function w = liblinearsvr(Data,C,normD)
    if normD == 2
        Data = normalizeL2(Data);
    end    
    if normD == 1
        Data = normalizeL1(Data);
    end    
    N = size(Data,1);
    Labels = [1:N]';
    model = train(double(Labels), sparse(double(Data)),sprintf('-c %1.6f -s 11 -q',C) );
    w = model.w';    
end

function Data = getNonLinearity(Data)
    Data = sign(Data).*sqrt(abs(Data));    
    %Data = vl_homkermap(Data',2,'kchi2');    
end

function x = normalizeL2(x)
    x=x./repmat(sqrt(sum(x.*conj(x),2)),1,size(x,2));
end
