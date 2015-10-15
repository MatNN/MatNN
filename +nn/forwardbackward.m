function [res, userRes] = forwardbackward(net, dzdy, res, opts, phase, usingGPU, userReq)
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
%  opts.accumulate      = false;
%  opts.outputBlobCount = cellfun(@numel, net.blobConnectId);

forget = opts.conserveMemory & (opts.learningRate==0);
ll = net.layers;
lo = net.layerobjs;
ww = net.weights;

if isempty(res)
    if usingGPU
        res.blob     = num2cell(gpuArray.zeros(1, numel(net.blobNames), 'single'));
        res.dzdx     = num2cell(gpuArray.zeros(1, numel(net.blobNames), 'single')); % each cell contains another cell, and the inner cell's length is respected to the number of bottoms that a layer accepts
        filler       = gpuArray(single(0));
    else
        res.blob     = num2cell(zeros(1, numel(net.blobNames), 'single'));
        res.dzdx     = num2cell(zeros(1, numel(net.blobNames), 'single')); % each cell contains another cell, and the inner cell's length is respected to the number of bottoms that a layer accepts
        %res.dzdw      = num2cell(zeros(1, numel(net.weightsNames), 'single')); % Each dzdw{w} corresponds to a net.weights{w}, no separate dzdw for each layer
        filler       = single(0);
    end
    res.dzdxCount    = zeros(1, numel(net.blobNames), 'single'); % count accumulation
    res.dzdwCount    = zeros(1, numel(net.weightsNames), 'single');
    res.dzdw         = cellfun(@(dd) dd.*single(0), net.weights, 'UniformOutput', false); % Each dzdw{w} corresponds to a net.weights{w}, no separate dzdw for each layer
    res.dzdwVisited  = false(size(res.dzdw));
else
    if isempty(res.blob)
        if usingGPU
            res.blob = num2cell(gpuArray.zeros(1, numel(net.blobNames), 'single'));
            res.dzdx = num2cell(gpuArray.zeros(1, numel(net.blobNames), 'single'));
            filler   = gpuArray(single(0));
        else
            res.blob = num2cell(zeros(1, numel(net.blobNames), 'single'));
            res.dzdx = num2cell(zeros(1, numel(net.blobNames), 'single'));
            filler   = single(0);
        end
        res.dzdxCount= zeros(1, numel(net.blobNames), 'single');
    end
end

res.dzdxCount   = res.dzdxCount.*0;
res.dzdwCount   = res.dzdwCount.*0;
res.dzdxVisited = false(size(res.dzdx)); % tops/bottoms ALWAYS DON'T NEED TO BE ACCUMULATED IF opts.accumulate SET TO 1.
                                         % ONLY SHARED tops NEED TO BE ACCUMULATED
                                         % SO WE RESET TO DALSe

for i = net.phase.(phase)
    l = ll{i};
  
    % if a layer don't generate output, it still should fill topBlob as {[],[],...}
    %if ~isempty(l.weights)
        weightsInd = l.weights(~net.weightsIsMisc(l.weights));
        miscInd = l.weights(net.weightsIsMisc(l.weights));
        [res.blob(l.top), ww(weightsInd), ww(miscInd)] = lo{i}.forward(opts, l, ww(weightsInd), ww(miscInd), res.blob(l.bottom), res.blob(l.top));
    %else
    %    [res.blob(l.top), ~] = net.layerobjs{i}.forward(opts, l, {}, res.blob(l.bottom));
    %end

    if ~isempty(userReq)
        for u=1:numel(userReq)
            userRes.(userReq{u}) = res.blob(net.blobNamesIndex.(userReq{u}));
        end
    else
        userRes = {};
    end

    % optionally forget intermediate results
    if forget && (~isfield(l, 'rememberOutput') || ~l.rememberOutput)
        for c = l.bottom
            co = opts.outputBlobCount(c);
            if co > 1
                opts.outputBlobCount(c) = opts.outputBlobCount(c)-1;
            elseif co == 1
                opts.outputBlobCount(c) = 0;
                res.blob(c) = {filler};
            elseif co == 0
                opts.outputBlobCount(c) = -1;
            end
        end
    end
end


% derivative
if opts.learningRate ~= 0

    % Make output blobs have their derivatives
    % consider the derivatives of all output blobs are
    % scalers, which are 1
    % You can make a weight scaler for loss, just write a
    % custom layer that multiplies the scaler onto it

    res.dzdx(opts.outputBlobCount==0) = {dzdy};
    
    for i = net.phase.(phase)(end:-1:1)
        l = ll{i};
        
        weightsInd = l.weights(~net.weightsIsMisc(l.weights));
        miscInd = l.weights(net.weightsIsMisc(l.weights));
        if opts.avgGradient
            tmpCount1 = res.dzdxCount(l.top);
            tmpCount1(tmpCount1==0) = 1;
            tmp_input_dzdx = res.dzdx(l.top);
            for yy = 1:numel(l.top)
                tmp_input_dzdx{yy} = tmp_input_dzdx{yy}./tmpCount1(yy);
            end
            tmpCount2 = res.dzdwCount(weightsInd);
            tmpCount2(tmpCount2==0) = 1;
            tmp_input_dzdw = res.dzdw(weightsInd);
            for yy = 1:numel(weightsInd)
                tmp_input_dzdw{yy} = tmp_input_dzdw{yy}./tmpCount2(yy);
            end
            [tmpdzdx, tmpdzdw, ww(miscInd)] = lo{i}.backward(opts, l, ww(weightsInd), ww(miscInd), res.blob(l.bottom), res.blob(l.top), tmp_input_dzdx, tmp_input_dzdw);
        else
            [tmpdzdx, tmpdzdw, ww(miscInd)] = lo{i}.backward(opts, l, ww(weightsInd), ww(miscInd), res.blob(l.bottom), res.blob(l.top), res.dzdx(l.top), res.dzdw(weightsInd));
        end
        
        
        % Don't try to clear res.dzdx or res.dzdw at first, you will get terrible performance!!
        % If you try to clear them at first so you can get rid of "if-statement" 
        % of opts.accumulate, the performance will drain a lot.
        
        dzdxEmpty = ~cellfun('isempty', tmpdzdx);

        for b = find(dzdxEmpty)
            %if any(net.blobConnectId.(phase){l.bottom(b)} == i) && ...
            %    ((~any(net.replaceId.(phase){l.bottom(b)} == i) && ~isempty(net.replaceId.(phase){l.bottom(b)})) || ...
            %    isempty(net.replaceId.(phase){l.bottom(b)})) && ...
            %    res.dzdxVisited(l.bottom(b))
            if any(net.blobConnectId.(phase){l.bottom(b)} == i) && ~any(net.replaceId.(phase){l.bottom(b)} == i) && res.dzdxVisited(l.bottom(b))
                res.dzdx{l.bottom(b)} = res.dzdx{l.bottom(b)} + tmpdzdx{b};
                res.dzdxCount(l.bottom(b)) = res.dzdxCount(l.bottom(b))+1;
            else
                res.dzdx(l.bottom(b)) = tmpdzdx(b);
                res.dzdxCount(l.bottom(b)) = 1;
            end
            res.dzdxVisited(l.bottom(b)) = true;
        end
        


        % be careful of modifying this.
        dzdwEmpty  = ~cellfun('isempty', tmpdzdw);
        for w = find(dzdwEmpty & res.dzdwVisited(weightsInd))
            res.dzdw{weightsInd(w)} = res.dzdw{weightsInd(w)} + tmpdzdw{w};
            res.dzdwCount(weightsInd(w)) = res.dzdwCount(weightsInd(w))+1;
        end

        dzdwEmpty2 = dzdwEmpty & ~res.dzdwVisited(weightsInd);
        res.dzdw(weightsInd(dzdwEmpty2)) = tmpdzdw(dzdwEmpty2);
        res.dzdwCount(weightsInd(dzdwEmpty2)) = 1;

        res.dzdwVisited(weightsInd) = true;
        % blow is slightly slower than loop (above)
        %res.dzdw(l.weights(dzdwEmpty1)) = cellfun(@plus, res.dzdw(l.weights(dzdwEmpty1)), tmpdzdw(dzdwEmpty1), 'UniformOutput', false);
        %res.dzdw(l.weights(dzdwEmpty2)) = tmpdzdw(dzdwEmpty2);
        %res.dzdwVisited(l.weights(dzdwEmpty)) = true;
    


        %if opts.conserveMemory %delete used dzdx{top}, no need to consider loss or accuracy, because der(loss)=1, and accuracy has no backward computation
        %    res.dzdx(l.top) = {filler};
        %end
        


        if strcmp(opts.backpropToLayer, l.name)
            break;
        end
    end
end
