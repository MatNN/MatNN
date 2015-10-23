function obj = StochasticGradientDescent(op, neto)
    gpufun = [];
    if numel(op.gpus)>0
        obj.solve = @solverGPUPTX;
        ptxp = [];
        cup = [];
        if isempty(ptxp)
            fileP = fileparts(mfilename('fullpath'));
            ptxp = fullfile(fileP, 'private', 'SGD.ptx');
            cup = fullfile(fileP, 'private', 'SGD.cu');
        end
        gpufun = nn.utils.gpu.createHandle(1, ptxp, cup, 'SGD');
        d = gpuDevice();
        MaxThreadsPerBlock = d.MaxThreadsPerBlock;
        gpufun.ThreadBlockSize = MaxThreadsPerBlock;
    else
        obj.solve = @solver;
    end
    clearvars neto;

    function net = solver(opts, lr, net, res, updateWeightsInd)
        m = opts.momentum;
        ts = opts.iter_size;
        for w = updateWeightsInd

            thisDecay = opts.weightDecay * net.weightDecay(w);
            thisLR = lr * net.learningRate(w);
            net.momentum{w} = m * net.momentum{w} - thisLR * (thisDecay*net.weights{w} + res.dzdw{w}/ts);
            net.weights{w}  = net.weights{w} + net.momentum{w};
            
            % CPU array does not support single value expand to an array,
            % so don't use this (only works for gpuArray)
            %[net.weights{w}, net.momentum{w}] = arrayfun(@mofun, opts.momentum, net.momentum{w}, thisLR, thisDecay, net.weights{w}, res.dzdw{w}/opts.iter_size);
        end
    end
    % function net = solverGPU(opts, lr, net, res, updateWeightsInd)
    %     m = single(opts.momentum);
    %     ts = single(opts.iter_size);
    %     wd = single(opts.weightDecay);
    %     for w = updateWeightsInd

    %         thisDecay = wd * net.weightDecay(w);
    %         thisLR = lr * net.learningRate(w);
            
    %         [net.weights{w}, net.momentum{w}] = arrayfun(@mofun, m, net.momentum{w}, thisLR, thisDecay, net.weights{w}, res.dzdw{w}, ts);
    %     end
    % end
    function net = solverGPUPTX(opts, lr, net, res, updateWeightsInd)
        m = single(opts.momentum);
        ts = single(opts.iter_size);
        wd = single(opts.weightDecay);
        for w = updateWeightsInd
            len = numel(net.weights{w});
            gpufun.GridSize = ceil( len/MaxThreadsPerBlock );
            [net.momentum{w}, net.weights{w}] = feval(gpufun, m, net.momentum{w}, lr, net.learningRate(w), wd, net.weightDecay(w), net.weights{w}, res.dzdw{w}, ts, len);
        end
    end

end

% function [w, mo1] = mofun(mo, mo1, lr, dc, w, dzdw, ts)
%     mo1  = mo.*mo1 - lr.*(dc.*w + dzdw/ts);
%     w = w+mo1;
% end