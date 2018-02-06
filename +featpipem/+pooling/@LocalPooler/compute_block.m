function pcode = compute_block(obj, assign, poolpos, locs, blk_count)
%COMPUTE_BLOCK Summary of this function goes here
%   Detailed explanation goes here

encode_len = obj.encoder_.get_output_dim();
pcode = zeros(encode_len, blk_count, 'single');

for iblk = 1:blk_count
  feats_sel = (locs == iblk);
  if any(feats_sel ~= 0)
    pcode(:,iblk) = featpipem.lib.mexPooler( double(assign), ...
      int32(poolpos), encode_len, obj.pool_type_i, feats_sel );
  else
    warning('CtxPool:EmptyBin', 'empty bin!');
  end
end

end

