function res = forwardbackward(net, x, dzdy, res, opts)
%FORWARDBACKWARD  Evaluates a neural network built by buildnet.m
%
%  Details:
%    Weights can be shared, if your weight_name set to the same name.
%    (eg. layers.weights)
%    Tops can be shared. (eg. forward of res)
%    Diffs cannot be shared. (eg. backward of res)
%
%
%  Default values: (for faster computation, disable value checking, you should
%                   provide all of the following options)
%
%  opts.accumulate = false;
%  opts.conserveMemory = false;
%  opts.sync = false;
%  opts.disableDropout = false;
%  opts.freezeDropout = false;
%  opts.visitLayerID = 1:numel(net.layers);
%  opts.outputBlobCount = cellfun(@numel, net.blobConnectId);
%  opts.gpuMode = false;
%  opts.doder = false;

forget = opts.conserveMemory & ~opts.doder;
ll = net.layers;
lo = net.layerobjs;
ww = net.weights;

if isempty(res)
    if opts.gpuMode
        res.blob  = num2cell(gpuArray.zeros(1, numel(net.blobNames), 'single'));
        res.dzdx  = num2cell(gpuArray.zeros(1, numel(net.blobNames), 'single')); % each cell contains another cell, and the inner cell's length is respected to the number of bottoms that a layer accepts
<<<<<<< HEAD

=======
        
>>>>>>> upstream/master
        filler    = gpuArray(single(0));
    else
        res.blob  = num2cell(zeros(1, numel(net.blobNames), 'single'));
        res.dzdx  = num2cell(zeros(1, numel(net.blobNames), 'single')); % each cell contains another cell, and the inner cell's length is respected to the number of bottoms that a layer accepts
        %res.dzdw  = num2cell(zeros(1, numel(net.weightsNames), 'single')); % Each dzdw{w} corresponds to a net.weights{w}, no separate dzdw for each layer
        filler    = single(0);
    end
    res.dzdw  = cellfun(@(dd) dd.*single(0), net.weights, 'UniformOutput', false); % Each dzdw{w} corresponds to a net.weights{w}, no separate dzdw for each layer
    res.dzdwVisited = false(size(res.dzdw));
end

res.dzdxVisited = false(size(res.dzdx));

for i = fieldnames(x)'
    name2Ind = net.blobNamesIndex.(i{1});
    if opts.gpuMode
        res.blob{name2Ind} = gpuArray(x.(i{1})); %Because x is a structure, eg. x = struct('data',[],'label',[])
    else
        res.blob{name2Ind} = x.(i{1}); %Because x is a structure, eg. x = struct('data',[],'label',[])
    end
end

for i = opts.visitLayerID
    l = ll{i};
<<<<<<< HEAD

=======
  
>>>>>>> upstream/master
    % if a layer don't generate output, it still should fill topBlob as {[],[],...}
    %if ~isempty(l.weights)
        weightsInd = l.weights(~net.weightsIsMisc(l.weights));
        miscInd = l.weights(net.weightsIsMisc(l.weights));
        [res.blob(l.top), ww(weightsInd), ww(miscInd)] = lo{i}.forward(opts, l, ww(weightsInd), ww(miscInd), res.blob(l.bottom), res.blob(l.top));
    %else
    %    [res.blob(l.top), ~] = net.layerobjs{i}.forward(opts, l, {}, res.blob(l.bottom));
    %end

    % optionally forget intermediate results
    if forget && (~isfield(l, 'rememberOutput') || ~l.rememberOutput)
        for c = l.bottom
            co = opts.outputBlobCount(c);
            if co > 1
                opts.outputBlobCount(c) = opts.outputBlobCount(c)-1;
            elseif co == 1
                opts.outputBlobCount(c) = 0;
                res.blob(l.bottom(c)) = filler;
            elseif co == 0
                opts.outputBlobCount(c) = -1;
            end
        end
    end
end


if opts.doder

    % Make output blobs have their derivatives
    % consider the derivatives of all output blobs are
    % scalers, which are 1
    % You can make a weight scaler for loss, just write a
    % custom layer that multiplies the scaler onto it

    res.dzdx(opts.outputBlobCount==0) = {dzdy};
<<<<<<< HEAD

    for i = opts.visitLayerID(end:-1:1)
        l = ll{i};

=======
    
    for i = opts.visitLayerID(end:-1:1)
        l = ll{i};
        
>>>>>>> upstream/master
        weightsInd = l.weights(~net.weightsIsMisc(l.weights));
        miscInd = l.weights(net.weightsIsMisc(l.weights));
        [tmpdzdx, res.dzdw(weightsInd), ww(miscInd)] = lo{i}.backward(opts, l, ww(weightsInd), ww(miscInd), res.blob(l.bottom), res.blob(l.top), res.dzdx(l.top), res.dzdw(weightsInd), res.dzdwVisited(weightsInd));
        res.dzdwVisited(weightsInd) = true;
        % Don't try to clear res.dzdx or res.dzdw at first, you will get terrble performace!!
        % If you try to clear them at first so you can get rid of if-statement of opts.accumulate
        % , the performance will drain a lot.
<<<<<<< HEAD

=======
        
>>>>>>> upstream/master
        dzdxEmpty = ~cellfun('isempty', tmpdzdx);

        for b = find(dzdxEmpty)
            if any(net.blobConnectId{l.bottom(b)} == i) && ((~any(net.replaceId{l.bottom(b)} == i) && ~isempty(net.replaceId{l.bottom(b)})) || isempty(net.replaceId{l.bottom(b)})) && res.dzdxVisited(l.bottom(b))
                res.dzdx{l.bottom(b)} = res.dzdx{l.bottom(b)} + tmpdzdx{b};
            else
                res.dzdx(l.bottom(b)) = tmpdzdx(b);
            end
            res.dzdxVisited(l.bottom(b)) = true;
        end
<<<<<<< HEAD
=======
        

>>>>>>> upstream/master


        % Legacy code
        % Layer functions should take care of weight sharing
        % this will be removed in the next update.
        %{
        % be careful of modifying this.
        dzdwEmpty = ~cellfun('isempty', tmpdzdw);
        dzdwEmpty2 = dzdwEmpty & ~res.dzdwVisited(l.weights);
        for w = find(dzdwEmpty & res.dzdwVisited(l.weights))
            res.dzdw{l.weights(w)} = res.dzdw{l.weights(w)} + tmpdzdw{w};
        end
        % blow is slightly slower than loop (above)
        %res.dzdw(l.weights(dzdwEmpty1)) = cellfun(@plus, res.dzdw(l.weights(dzdwEmpty1)), tmpdzdw(dzdwEmpty1), 'UniformOutput', false);
        res.dzdw(l.weights(dzdwEmpty2)) = tmpdzdw(dzdwEmpty2);
        res.dzdwVisited(l.weights(dzdwEmpty)) = true;
        %}
<<<<<<< HEAD

=======
    
>>>>>>> upstream/master
        if opts.conserveMemory %delete used dzdx{top}, no need to consider loss or accuracy, because der(loss)=1, and accuracy has no backward computation
            res.dzdx(l.top) = {filler};
        end
    end
end
