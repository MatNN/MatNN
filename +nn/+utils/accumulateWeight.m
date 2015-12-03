function accumulateWeight(net, weightsInd, varargin)
% be careful of modifying this.

dzdwEmpty  = ~cellfun('isempty', varargin); % find d_weights are not empty
for w = find(dzdwEmpty & net.weightsDiffCount(weightsInd))
    net.weightsDiff{weightsInd(w)} = net.weightsDiff{weightsInd(w)} + varargin{w};
    net.weightsDiffCount(weightsInd(w)) = net.weightsDiffCount(weightsInd(w))+1;
    %fprintf('Weight %s accumulated %d times!\n', net.weightsNames{weightsInd(w)}, net.weightsDiffCount(weightsInd(w)));
end

dzdwEmpty2 = dzdwEmpty & ~net.weightsDiffCount(weightsInd);
net.weightsDiff(weightsInd(dzdwEmpty2)) = varargin(dzdwEmpty2);
net.weightsDiffCount(weightsInd(dzdwEmpty2)) = 1;

end