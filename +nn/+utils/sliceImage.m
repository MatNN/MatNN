function [sliced4DTensor] = sliceImage(image4DTensor, sliceSize, pad)
%SLICEIMAGE slice images and add padding for you
% sliceSize = [height width] or one value
%             sliced image size (without overlapping)
% pad = [TOP BOTTOM LEFT RIGHT] or one value
%       (padding will be add before slice!)
%
% NOTICE
% pad > 0 also means each sliced image will overlapped with size 'pad'
%
% if size(image4DTensor) == [H,W,C,N]
% then size(sliced4DTensor) == [sliceSize(1)+pad(1)+pad(2), sliceSize(2)+pad(3)+pad(4), C, N*(H/sliceSize(1) * W/sliceSize(2))]


[H, W, C, N] = size(image4DTensor);
if numel(sliceSize) == 1
    sliceSize(2) = sliceSize;
end
if numel(pad) == 1
    pad = [pad, pad, pad, pad];
end

if mod(H,sliceSize(1))~=0 || mod(W,sliceSize(2))~=0
    error('Currently only support sliceSize which is a factor of image size.');
end

pad4DTensor = zeros(H+pad(1)+pad(2), W+pad(3)+pad(4), C, N, 'single');
H_ = H+pad(1)+pad(2);
W_ = W+pad(3)+pad(4);

% add padding
pad4DTensor(pad(1)+1:pad(1)+H, pad(3)+1:pad(3)+W, :, :) = image4DTensor;

% create sliceIndex
indH_  = pad(1)+pad(2):sliceSize(1):H_;
indH = 1:sliceSize(1):H;

indW_  = pad(3)+pad(4):sliceSize(2):W_;
indW = 1:sliceSize(2):W;


sliced4DTensor = zeros(sliceSize(1)+pad(1)+pad(2), sliceSize(2)+pad(3)+pad(4), C, N*(H/sliceSize(1) * W/sliceSize(2)), 'single');
for j=1:numel(indW)
    for i=1:numel(indH)
        ind1 = (j-1)*H/sliceSize(1)+i;
        sliced4DTensor(:,:,:,N*(ind1-1)+1:N*ind1) = pad4DTensor(indH(i):indH_(i+1), indW(j):indW_(j+1), :, :);
    end
end

end
