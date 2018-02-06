function pcode = compute(obj, im)
%COMPUTE Pool features using the spatial pyramid match kernel

if isstruct(im)
   vid_size = im.vid_size;
   im = im.path;
   [feats,frames] = obj.featextr_.compute(im);
   [assign,poolpos] = obj.encoder_.encode(feats);
   pcode = obj.compute_impl(vid_size, assign, poolpos, frames);
else
    [feats,frames] = obj.featextr_.compute(im);
    [assign,poolpos] = obj.encoder_.encode(feats);
    % ���� VLAD-K ������һЩ���⣬��ͷȷ��һ�£�Ŀǰ k ��ά��ֱ�Ӵ���
    if isa(im,'float') && size(im,3)<=3
        pcode = obj.compute_impl([size(im,1) size(im,2) 0], assign, poolpos, frames);
        % as image doesn't have a temporal dimension, set it 0   
    else
%         pcode = obj.compute_impl(frames(:,end)', assign, poolpos, frames(:,1:end-1));
    end
end



pcode = obj.compute_rp(pcode);

end