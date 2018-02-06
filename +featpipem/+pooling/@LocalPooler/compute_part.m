function pcode = compute_part(obj, assign, poolpos, locs, anchors, radius)
%COMPUTE_PART Summary of this function goes here
%   Detailed explanation goes here

bottom = anchors - radius;
top = anchors + radius;

blk_count = size(anchors, 2);
encode_len = obj.encoder_.get_output_dim();
pcode = zeros(encode_len, blk_count, 'single');

for iblk = 1:blk_count
  feats_sel = all(locs-bottom(:,iblk) >= 0) & all(top(:,iblk)-locs >= 0);
  if any(feats_sel ~= 0)
    pcode(:,iblk) = featpipem.lib.mexPooler( double(assign), ...
      int32(poolpos), encode_len, obj.pool_type_i, feats_sel );
  else
    warning('CtxPool:EmptyBin', 'empty bin!');
  end
end

end

