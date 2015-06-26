function [slicedVector, slicedInd] = sliceVector(AVector, parts)
%SLICEVECTOR
%  this function slice AVector into parts
%  INPUT
%  AVector      A vecotor
%  parts        A number of sliced parts
%  OUTPUT
%  slicedVector A 2D-matrix, each row is a part of AVector, row number == parts

if numel(AVector) < parts
    error('Vector length must >= parts');
end

% eg. 1~10, parts=2.5
parts = numel(AVector)/parts;
slicedInd1 = ceil(1:parts:numel(AVector)); % 1  3.5    6   8.5      -> 1:4:6:9
slicedInd2 = [slicedInd1(2:end)-1,numel(AVector)]; % 1  3.5    6   8.5      -> 0,3,5,8,10


slicedInd = [slicedInd1; slicedInd2]';
slicedVector = AVector(slicedInd);

end
