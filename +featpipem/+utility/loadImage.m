function im = loadImage( impath )
%LOADIMAGE Summary of this function goes here
%   Detailed explanation goes here

[im,clmap] = imread(impath);
% gif, use the first frame
if ndims(im) == 4
  im = im(:,:,:,1);
end
if ~isempty(clmap)
  im = ind2rgb(im, clmap);
end

end

