function res = forwardbackward(net, x, dzdy, res, opts)
%FORWARDBACKWARD  Evaluates a neural network built with buildnet.m
%
%
%  Forward:
%    'res' is a structure, each field is the tops of layers
%  Backward:
%    'res' is a cell array, each cell is corresponds to a specific layer
%
%  Details:
%    Weights can be shared, if your weight_name set to the same name.
%    (eg. layers.weights)
%    Tops can be shared. (eg. forward of res)
%    Diffs cannot be shared. (eg. backward of res)
%
%  NOTICE
%  if your layer produces .misc, you need to maintain its gpu/cpu array consistency.
%
%  Default values: (for faster computation, disable value checking, you should
%                   provide all of the following options)
%
%  
%  opts.accumulate = false;
%  opts.conserveMemory = false;
%  opts.sync = false;
%  opts.disableDropout = false;
%  opts.freezeDropout = false;
%  opts.visitLayerID = 1:numel(net.layers);
%  opts.gpuMode = false;
%  opts.doder = false;


if isempty(res)
    res.blob  = num2cell(zeros(1, numel(net.blobNames), 'single'));
    res.dzdx  = num2cell(zeros(1, numel(net.blobNames), 'single')); % each cell contains another cell, and the inner cell's length is respected to the number of bottoms that a layer accepts
    res.dzdw  = num2cell(zeros(1, numel(net.weightsNames), 'single')); % Each dzdw{w} corresponds to a net.weights{w}, no separate dzdw for each layer
    res.dzdwVisited = false(size(res.dzdw));
end

for i = fieldnames(x)'
    name2Ind = net.blobNamesIndex.(i{1});
    res.blob{name2Ind} = x.(i{1}); %Because x is a structure, eg. x = struct('data',[],'label',[])
end

for i = opts.visitLayerID
    l = net.layers{i};
    lo = net.layerobjs{i};
  
    [topBlob, weightUpdate] = lo.forward(opts, l, net.weights(l.weights), res.blob(l.bottom));
    res.blob(l.top) = topBlob; % if a layer don't generate output, it still should fill topBlob as {[],[],...}
  
    if ~isempty(weightUpdate)
        net.weights(l.weights(weightUpdate{1})) = weightUpdate{2};
    end
  
    % optionally forget intermediate results
    forget = opts.conserveMemory;
    forget = forget & (~opts.doder || strcmp(lo.name, 'ReLU'));
    forget = forget & ~lo.generateLoss;
    forget = forget & (~isfield(l, 'rememberOutput') || ~l.rememberOutput);
    if forget
        res.blob(l.top) = {[]};
    end
    if opts.gpuMode && opts.sync
        % This should make things slower, but on MATLAB 2014a it is necessary
        % for any decent performance.
        wait(gpuDevice);
    end
end


if opts.doder

    % Make output blobs have their derivatives
    % consider the derivatives of all output blobs are
    % scalers, which are 1
    % You can make a weight scaler for loss, just write a
    % custom layer that multiplies the scaler onto it
    outputBlob = cellfun('isempty', net.blobConnectId);
    res.dzdx(outputBlob) = {dzdy};
  
    for i = opts.visitLayerID(end:-1:1)
        l = net.layers{i};
    
        [tmpdzdx, tmpdzdw] = net.layerobjs{i}.backward(opts, l, net.weights(l.weights), res.blob(l.bottom), res.dzdx(l.top));
        

        % Don't try to clear res.dzdx or res.dzdw at first, you will get terrble performace!!
        % If you try to clear them at first so you can get rid of if-statement of opts.accumulate
        % , the performance will drain a lot.
        dzdxEmpty = ~cellfun('isempty', tmpdzdx);
        if opts.accumulate
            for b = find(dzdxEmpty)
                if any(net.blobConnectId(l.bottom(b)) == i)
                    res.dzdx{l.bottom(b)} = res.dzdx{l.bottom(b)} + tmpdzdx{b};
                else
                    res.dzdx(l.bottom(b)) = tmpdzdx(b);
                end
            end
        else
            res.dzdx(l.bottom(dzdxEmpty)) = tmpdzdx(dzdxEmpty);
        end
        
        % be careful of modifying this.
        dzdwEmpty = ~cellfun('isempty', tmpdzdw);
        st = res.dzdwVisited(l.weights) | opts.accumulate;
        %dzdwEmpty1 = dzdwEmpty & st;
        dzdwEmpty2 = dzdwEmpty & ~st;
        for w = find(dzdwEmpty & st)
            res.dzdw{l.weights(w)} = res.dzdw{l.weights(w)} + tmpdzdw{w};
        end
        % blow is slightly slower than loop (above)
        %res.dzdw(l.weights(dzdwEmpty1)) = cellfun(@plus, res.dzdw(l.weights(dzdwEmpty1)), tmpdzdw(dzdwEmpty1), 'UniformOutput', false);
        res.dzdw(l.weights(dzdwEmpty2)) = tmpdzdw(dzdwEmpty2);
    
        res.dzdwVisited(l.weights(dzdwEmpty)) = true;
    
        if opts.conserveMemory %delete used dzdx{top}, no need to consider loss or accuracy, because der(loss)=1, and accuracy has no backward computation
            res.dzdx(l.top) = {[]};
        end
        if opts.gpuMode && opts.sync
            wait(gpuDevice);
        end
    end
end
