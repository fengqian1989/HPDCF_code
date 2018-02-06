function [pcode,params] = compute_node(obj, assign, poolpos, locs, nodes)
%COMPUTE_NODE Summary of this function goes here
%   Detailed explanation goes here

bin_count = size(nodes, 2);
encode_len = obj.encoder_.get_output_dim();
pcode = zeros(encode_len, bin_count, 'single');

dist = pdist2(single(locs)', single(nodes)');
for ibin = 1:bin_count
  [~,feats_sel] = sort(dist(:,ibin), 'ascend');
  feats_sel = int32( feats_sel(1:ceil(end/bin_count)) )';
  pcode(:,ibin) = featpipem.lib.mexPooler( assign, ...
    int32(poolpos), encode_len, obj.pool_type_i, feats_sel );
end

params.bin_dist = dist;

% dist = pdist2(nodes', single(locs)');
% [~,blkassign] = min(dist, [], 1);
% for iblk = 1:blk_count
%   feats_sel = (blkassign == iblk);
%   if any(feats_sel ~= 0)
%     pcode(:,iblk) = featpipem.lib.mexPooler( double(assign), ...
%       int32(poolpos), encode_len, obj.pool_type_i, feats_sel );
%   else
%     warning('CtxPool:EmptyBin', 'empty bin!');
%   end
% end

end

