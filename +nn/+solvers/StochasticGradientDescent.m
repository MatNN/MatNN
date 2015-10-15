function obj = StochasticGradientDescent(architecture, neto)

    obj.solve = @solver;
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

end

function [w, mo1] = mofun(mo, mo1, lr, dc, w, dzdw)
    mo1  = mo.*mo1 - lr.*(dc.*w + dzdw);
    w = w+mo1;
end