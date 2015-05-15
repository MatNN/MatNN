function obj = StochasticGradientDescent()

    obj.setup = @setup;
    obj.solve = @solver;
    cuKernel = [];

    function solve = setup(architecture)
        if strcmp(architecture, 'cuda kernel')
            solve = @solver;
            ptxFile = fullfile('+nn','+solvers','StochasticGradientDescent.ptx');
            cuFile = fullfile('+nn','+solvers','StochasticGradientDescent.cu');

            % setup forward kernel
            cuKernel = parallel.gpu.CUDAKernel(ptxFile, cuFile, 'forward');
            cuKernel.ThreadBlockSize = cuKernel.MaxThreadsPerBlock;
            %cuKernel.GridSize = [];
        end
    end
    function net = solver(opts, lr, batchSize, net, res)
        for w = 1:numel(res.dzdw)
            %There are 3 cases
            % 1. layer1 -> layer2 -> ...
            %    just compute corresponds dzdw
            % 2. layer1 -> {layer2
            %           -> {layer3
            %    sum up all dzdw of layer2 and layer3
            % 3. layer1} -> layer3
            %    layer2} -^
            %    just like 1.
            % 4. if above case involves share weights
            % 4-1. layer1 -> layer1' -> ...
            %      ALLOWED, but do this at users own risk.
            % 4-2. layer1 -> {layer2'
            %             -> {layer2''
            %      just use 2.
            % 4-3. layer1 } -> layer2
            %      layer1'} -^
            %      sum up gradient!!!
            %
            % Solution:
            % 1.2.3. solved by simplenn, direct add dzdx to corresponds top's dzdx
            %        becuase currently not yet implemnt a layer with wights+ multiple tops
            %        so need to verify 2. !!!!!!!!!
            % 4-2.   same as 2.
            % 4-3.   solved by the scheme of separate weights/momentum from net.layers
            %        so weights will be update twice (WRONG!!!!!)
            %        Need to sum up gradient!!!!!!! do this now!!!
            %        SOLVED!!!!, no need to no extra works.
            % 4-1.   same as 1.
            %
            %{
            thisDecay = opts.weightDecay * net.weightDecay(w) ;
            thisLR = lr * net.learningRate(w) ;
            net.momentum{w} = ...
              opts.momentum * net.momentum{w} ...
              - thisDecay * net.weights{w} ...
              - (1 / batchSize) * res.dzdw{w} ;
            net.weights{w} = net.weights{w} + thisLR * net.momentum{w} ;
            %}
            thisDecay = opts.weightDecay * net.weightDecay(w) ;
            thisLR = lr * net.learningRate(w) / batchSize ;
            [net.weights{w}, net.momentum{w}] = arrayfun(@mofun, opts.momentum, net.momentum{w}, thisLR, thisDecay, net.weights{w}, res.dzdw{w});
        end

    end
    function net = solver_CUDAKernel(opts, lr, batchSize, net, res)
        for w = 1:numel(res.dzdw)
            thisDecay = opts.weightDecay * net.weightDecay(w) ;
            thisLR = lr * net.learningRate(w) / batchSize ;
            [net.weights{w}, net.momentum{w}] = feval(cuKernel, net.weights{w}, net.momentum{w}, thisLR, thisDecay, opts.momentum, res.dzdw{w}, numel(net.weights{w}));
        end
        
    end


end

function [w, mo1] = mofun(mo, mo1,lr,dc,w,dzdw)
    mo1  = mo.*mo1 - lr.*(dc.*w + dzdw);
    w = w+mo1;
end