function o = nn(netObj)
%NN  (Based on the example code of Matconvnet)
%
%  NOTE
%    provided 'batchStructTest' will fetch all samples and test them one by one
%


o.init = @init;
o.initParameter = @initParameter;
o.run = @runLoop;


% Init parameters
op   = {};
opts = {};
net  = [];
userOutputs = [];
userRequest = {};

% ================== NESTED FUNCTION DEFINITION SECTION ===================
% -------------------------------------------------------------------------
%                                 give desired variable name and get result
% -------------------------------------------------------------------------
    function [trained_net, selected_outputs] = runLoop(varargin)

        assert(~isempty(op),   'You must run init() to establish options for your network.');
        assert(~isempty(opts), 'You must run initParameter() to establish options for each phase.');

        % Init network
        %[net, numGpus, evaluateMode]  = initNetwork();
        initNetwork();

        %check if your requests is existed
        if nargin>0
            assert(all(ismember(varargin, net.blobNames)), 'Some of your requested blob names are not in your network definition.');
            userRequest = varargin;
        end

        % Start train / test
        loop();
        
        trained_net = net;

        selected_outputs = userOutputs;
    end

% -------------------------------------------------------------------------
%                                           Oracle Parameter initialization
% -------------------------------------------------------------------------
    function init(op_user)
        op      = {};
        opts    = {};
        net     = [];

        op.phaseOrder     = {'train', 'test'};
        op.repeatTimes    = 10;        % opts.numToNext * op.repeatTimes = total iterations
        op.continue       = [];        % Set to <iter> to load specific intermediate model. eg. 300, 10, 36000
        op.expDir         = fullfile('data','exp');
        op.gpus           = [];
        
        op = vl_argparse(op, op_user);

        if ~exist(op.expDir, 'dir')
            mkdir(op.expDir);
        end
    end

% -------------------------------------------------------------------------
%                                                  Parameter initialization
% -------------------------------------------------------------------------
    function initParameter(opt_user, phase)
        opt.numToNext          = 100;   % Runs how many iterations to next phase
        opt.numToSave          = 50;    % Runs how many iterations to next save intermediate model
        opt.displayIter        = 10;    % Show info every opt.displayIter iterations
        opt.showFirstIter      = true;  % show first iteration info
        opt.iter_size          = 1;     % number of iterations to accumulate gradients and update weights.
                                        % useful for divide a batch into multiple subbatches (to fit limited memory capacity)

        opt.learningRate       = 0.001; % 0 = no backpropagation
        opt.learningRateGamma  = 0.1;
        opt.learningRatePower  = 0.75;
        opt.learningRateSteps  = 1000;
        opt.learningRatePolicy = @(currentTotalIterNumber, currentPhaseTotalIter, lr, gamma, power, steps) lr*(gamma^floor(currentPhaseTotalIter/steps));
        
        opt.weightDecay        = 0.0005;
        opt.momentum           = 0.9;

        opt.conserveMemory     = false; % true: Delete forward results at each iteration, but runs slightly slower
        
        opt.solver             = @nn.solvers.StochasticGradientDescent;
        opt.avgGradient        = false; % if true, accumulated gradient will be divided by .iter_size and shared counts.
        opt.backpropToLayer    = []; % set this to a layer name, if you want to back propagate up to the specified layer.


        if isfield(opt_user, 'layerSettings')
            opt.layerSettings = opt_user.layerSettings; % you can design your own options for custom layer
        end

        opt.plotDiagnostics = false;
        opt = vl_argparse(opt, opt_user);

        if isempty(opt.numToNext)
            error('You must set opt.numToNext.');
        end


        % set opts
        opts.(phase) = opt;
    end

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------
    function initNetwork()
        % setup GPUs
        numGpus = numel(op.gpus);
        if numGpus > 1
            if isempty(gcp('nocreate')),
                parpool('local', numGpus);
                spmd
                    gpuDevice(op.gpus(labindex)) % print GPU info
                end
            end
        elseif numGpus == 1
            gpuDevice(op.gpus)  % print GPU info
        end
        
        net = netObj.getNet(op);
        netObj.clear();
        clearvars netObj;
    end

% -------------------------------------------------------------------------
%                                                       Loop for all phases
% -------------------------------------------------------------------------
    function loop()
        numGpus  = numel(op.gpus);


        % Load intermediate savings
        netName = net.name; % to prevent memory consuming problem, save net name to another variable
        modelPath = @(ep) fullfile(op.expDir, sprintf('%s-Iter%d.mat', netName, ep));
        if ~isempty(op.continue)
            if exist(modelPath(op.continue),'file')
                %if op.continue == op.repeatTimes*opts.(LoadPhase).numToNext
                %    fprintf('Load all phase process completed network: %s\n', modelPath(opt.continue));
                %else
                    fprintf('Resuming by loading Iter%d\n', op.continue);
                %end

                % temporarily disable warning message of loading CUDA kernels
                warnStruct = warning('off','all');
                tmp = load(modelPath(op.continue), 'net');
                warning(warnStruct);
                
                % Consider that loaded net is the same as our net, so just use their
                % weights
                %netObj.setBaseNet(tmp.net);
                %net = netObj.getNet();

                net.weights = tmp.net.weights;
                net.momentum = tmp.net.momentum;

                clearvars tmp;
            else
                error('Can''t find savings.');
            end
        end

        % move to cpu or gpu
        if numGpus > 0
            net = nn.utils.movenet(net, 'gpu');
        end

        % Calculate current phase, phase iteration
        onceRunIterNum = 0;
        for f = op.phaseOrder
            onceRunIterNum = onceRunIterNum + opts.(f{1}).numToNext;
        end
        if ~isempty(op.continue)
            currentTotalIter = op.continue+1; % continued from saved (iter+1)
        else
            currentTotalIter = 1;
        end
        iter = floor(currentTotalIter/onceRunIterNum)+1;
        modIter = mod(currentTotalIter-1, onceRunIterNum)+1;
        % Find current phase and current phase iter
        for f = 1:numel(op.phaseOrder)
            if modIter > opts.(op.phaseOrder{f}).numToNext
                modIter = modIter-opts.(op.phaseOrder{f}).numToNext;
            else
                currentFace = f;
                currentIter = modIter;
                break;
            end
        end
        clearvars modIter;

        % Start main loop
        startTime = tic;
        firstLoad = true;
        tmpHandle = @runPhase;
        globalIterNum = currentTotalIter;
        for i = iter:op.repeatTimes
            if firstLoad
                f = currentFace;
                firstLoad = false;
            else
                f = 1;
            end
            
            while(f<=numel(op.phaseOrder))
                disp('==========================================================================');
                if numGpus <= 1
                    runPhase(op.phaseOrder{f}, i, globalIterNum, currentIter, modelPath);
                else
                    spmd(numGpus)
                        % Nested function handle doesn't have parent workspace, so we need to return 'net' variable
                        net_multiGPU = tmpHandle(op.phaseOrder{f}, i, globalIterNum, currentIter, modelPath);
                    end
                    net = net_multiGPU{1};
                end
                globalIterNum = globalIterNum + opts.(op.phaseOrder{f}).numToNext;
                f = f+1;
            end
            currentIter = 1;
        end

        % Report training/evaluation time
        startTime = toc(startTime);
        fprintf('Total running time: %.2fs\n', startTime);
    end

% -------------------------------------------------------------------------
%                                                     Core function of loop
% -------------------------------------------------------------------------
    function [varargout] = runPhase(currentFace, currentRepeatTimes, globalIterNum, currentIter, modelPath)
        numGpus  = numel(op.gpus);
        outputBlobID = net.outputBlobId.(currentFace);
        sourceBlobID = net.sourceBlobId.(currentFace);
        
        
        if numGpus >= 1
            one = gpuArray(single(1));
        else
            one = single(1);
        end

        optface = opts.(currentFace);

        % Create options for forwardbackward
        optface.disableDropout  = optface.learningRate == 0;
%        optface.freezeDropout   = false;
        optface.outputBlobCount = cellfun(@numel, net.blobConnectId.(currentFace));
        optface.name = currentFace;
        optface.gpuMode = numGpus >= 1;


        % Find initial weight learning rate ~= 0 to update them
        needToUpdatedWeightsInd = find(~net.weightsIsMisc & ~cellfun(@isempty,net.weights));
        if ~isempty(optface.backpropToLayer)
            nw = [];
            for ww = numel(net.phase.(currentFace)):-1:1
                l = net.phase.(currentFace){ww};
                if isfield(l, 'weights')
                    nw = [nw, l.weights]; %#ok
                end
                if strcmp(optface.backpropToLayer, l.name)
                    break;
                end
            end
            [~,wind] = setdiff(needToUpdatedWeightsInd,nw);
            needToUpdatedWeightsInd(wind) = [];
        end

        % Calculate total iteration number, current phase total iteration number
        currentPhaseTotalIter = (currentRepeatTimes-1)*optface.numToNext+currentIter;
        sover = optface.solver(net);

        accumulateOutBlobs = zeros(size(outputBlobID));
        res = [];
        count = 1;count_per_display = 1;
        phaseTime = tic;
        for t = currentIter:optface.numToNext
            % set learning rate
            learningRate = optface.learningRatePolicy(globalIterNum, currentPhaseTotalIter, optface.learningRate, optface.learningRateGamma, optface.learningRatePower, optface.learningRateSteps);

            % set currentIter
            optface.currentIter = t;
            
            % run subbatches
            iterOutBlobs = zeros(size(outputBlobID));

            % conserveMemory
            if optface.conserveMemory
                res = [];
            end
            for s=1:optface.iter_size
                % evaluate CNN
                if optface.learningRate > 0
                    dzdy = one;
                else
                    dzdy = [];
                end
                optface.accumulate = s > 1;
                [res, userOutputs] = nn.forwardbackward(net, dzdy, res, optface, currentFace, numGpus >= 1, userRequest);

                % accumulate backprop errors
                % assume all output blobs are loss-like blobs
                for ac = 1:numel(accumulateOutBlobs)
                    blobRes = double(gather( res.blob{outputBlobID(ac)} ));
                    iterOutBlobs(ac) = iterOutBlobs(ac) + sum(blobRes(:));
                    accumulateOutBlobs(ac) = accumulateOutBlobs(ac) + sum(blobRes(:));
                end
                if optface.conserveMemory
                    res.blob = {};
                    res.dzdx = {};
                end
            end
            res.dzdwVisited = res.dzdwVisited & false;

            if optface.learningRate ~= 0
                if numGpus <= 1
                    net = sover.solve(optface, learningRate, net, res, needToUpdatedWeightsInd);
                else
                    labBarrier();
                    %accumulate weight gradients from other labs
                    res.dzdw = gop(@(a,b) cellfun(@plus, a,b, 'UniformOutput', false), res.dzdw);
                    net = sover.solve(optface, learningRate, net, res, needToUpdatedWeightsInd);
                end
            end


            % Print learning statistics
            if mod(count, optface.displayIter) == 0 || (count == 1 && optface.showFirstIter)
                if optface.learningRate ~= 0
                    preStr = [datestr(now, '[mmdd HH:MM:SS.FFF '), sprintf('Lab%d—%s] fi%d/gi%d, lr = %g, ', labindex, currentFace,currentPhaseTotalIter, globalIterNum, learningRate)];
                else
                    preStr = [datestr(now, '[mmdd HH:MM:SS.FFF '), sprintf('Lab%d—%s] fi%d/gi%d, ', labindex, currentFace,currentPhaseTotalIter, globalIterNum)];
                end
                
                for ac = 1:numel(accumulateOutBlobs)
                    if isinf(accumulateOutBlobs(ac))
                        fprintf('\n');
                        error('A blob output = Inf');
                    elseif ~isempty(accumulateOutBlobs(ac))
                        fprintf(preStr);
                        fprintf('%s = %.6g ', net.blobNames{outputBlobID(ac)}, accumulateOutBlobs(ac)./(optface.iter_size*count_per_display)); % this is a per-batch avg., not output avg.
                    end
                    if ac~=numel(accumulateOutBlobs)
                        fprintf('\n');
                    end
                end

                fprintf('%.2fs(%.2f iter/s)\n', toc(phaseTime), count_per_display/toc(phaseTime));
                phaseTime = tic;
                accumulateOutBlobs = zeros(size(outputBlobID));
                count_per_display = 0;
            end

            % Save model
            if ~isempty(optface.numToSave) && mod(count, optface.numToSave) == 0
                if numGpus > 1
                    labBarrier();
                end
                if labindex == 1 % only one worker can save the model
                    fprintf('Saving network model to %s ... \n', modelPath(globalIterNum));
                    net_back = net;
                    net = nn.utils.movenet(net, 'cpu');
                    save(modelPath(globalIterNum), 'net');
                    net = net_back;
                end
            end

            count = count+1;
            count_per_display = count_per_display+1;
            globalIterNum = globalIterNum+1;
            currentPhaseTotalIter = currentPhaseTotalIter+1;

        end

        % Set output for spmd
        varargout{1} = net;
    end % function end

% End of main function
end
