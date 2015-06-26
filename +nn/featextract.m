function [net, outBlob] = featextract(netObj, outBlobNames, batchStruct, opts_user)
%TRAIN  (Based on the example code of Matconvnet)
%
%  NOTE
%    provided 'batchStructTest' will fetch all samples and test them one by one
%

opts.numEpochs = [];
opts.numInterations = [];
opts.numToTest = 1; %runs how many Epochs or iterations to test
opts.numToSave = 10; % note, this value must be dvidable by opts.numToTest
opts.displayIter = 10; %Show info every opts.displayIter iterations
opts.batchSize = 256;
opts.numSubBatches = 1;
opts.gpus = [];
opts.computeMode = 'default';

opts.learningRate = 0.001;
opts.learningRateGamma = 0.1;
opts.learningRatePower = 0.75;
opts.learningRateSteps = 1000;
opts.learningRatePolicy = @(currentBatchNumber, lr, gamma, power, steps) lr*(gamma^floor(currentBatchNumber/steps));
opts.weightDecay = 0.0005;
opts.momentum = 0.9;
opts.solver   = @nn.solvers.StochasticGradientDescent;

opts.continue = [] ; % if you specify the saving's iteration/epoch number, then you can load it
opts.expDir = fullfile('data','exp');
opts.conserveMemory = false;
opts.sync = false;
opts.prefetch = false;

opts.plotDiagnostics = false;
opts = vl_argparse(opts, opts_user);

if isempty(batchStruct), error('Must specify Input batch struct!!'); end
if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir); end


% opts.numEpochs and opts.numInterations cannot be set at the same time
if isempty(opts.numEpochs) + isempty(opts.numInterations) == 1
    if ~isempty(opts.numEpochs)
        opts.epit = 'Epoch'; %epit means epoch_iteration
    else
        opts.epit = 'Iter';
    end
else
    error('You must set opts.numEpochs OR opts.numInterations.');
end

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

evaluateMode = true;

% setup GPUs
numGpus = numel(opts.gpus) ;
if numGpus > 1
    if isempty(gcp('nocreate')),
        parpool('local',numGpus) ;
        spmd, gpuDevice(opts.gpus(labindex)), end
    end
elseif numGpus == 1
    gpuDevice(opts.gpus)
end

if numGpus == 0 && ~strcmp(opts.computeMode, 'default')
    opts.computeMode = 'default';
    warning('Set computeMode to ''default''.');
end
netObj.computeMode(opts.computeMode);
net = netObj.getNet();
disp(['Set to <strong>', opts.computeMode, '</strong> mode']);

% -------------------------------------------------------------------------
%                                           Find train/test phase layer ID
% -------------------------------------------------------------------------

visitLayerID = [];
for i=1:numel(net.layers)
    visitLayerID = [visitLayerID, i]; %#ok
end

% find the index of output blob for feature-extraction
outputBlobID = [];
for i=1:numel(outBlobNames)
    ob = getfield(net.blobNamesIndex, outBlobNames{i});
    outputBlobID = [outputBlobID, ob];
end

% initialize some variable for this function
trainPa.branchedTopID = find(cellfun(@numel, net.blobConnectId) > 1);


% -------------------------------------------------------------------------
%                                                            Train and test
% -------------------------------------------------------------------------

%Load exist tmp savings
modelPath = @(ep) fullfile(opts.expDir, sprintf('%s-%s%d.mat', net.name, lower(opts.epit), ep));
if ~isempty(opts.continue)
    if exist(modelPath(opts.continue),'file')
        if opts.continue == runTimes
            fprintf('Load completed training network: %s\n', modelPath(opts.continue)) ;
        else
            fprintf('Resuming by loading %s%d\n', lower(opts.epit), opts.continue) ;
        end
        tmp = load(modelPath(opts.continue), 'net') ;

        % Consider that loaded net is the same as our net, so just use their
        % weights
        %netObj.setBaseNet(tmp.net);
        %net = netObj.getNet();

        net.weights = tmp.net.weights;
        net.momentum = tmp.net.momentum;
        clearvars tmp;
    else
        error('Can''t found savings.');
    end
end

% Save current time
startTime = tic;

% move CNN to GPU as needed
if numGpus == 1
    net = nn.utils.movenet(net, 'gpu') ;
elseif numGpus > 1
    spmd(numGpus)
        net_ = nn.utils.movenet(net, 'gpu') ;
    end
end

% train one epoch (or achieved opts.numToTest) and test
% process_runs msut accept startIndex and endIndex of iter/epoch
if numGpus <= 1
    %generate random seed
    %rng(randomSeed);
[net, outBlob] = process_runs(opts, numGpus, net, batchStruct, visitLayerID, outputBlobID, 1);
else
    spmd(numGpus)
        %generate random seed
        %rng(randomSeed);
        [net_, outBlob] = process_runs(opts, numGpus, net, batchStruct, visitLayerID, outputBlobID, 1);
    end
end

%restores rng state
%rng(rngState);

% Report training/evaluation time
startTime = toc(startTime);
fprintf('<strong>Total running time:</strong> %.2fs\n', startTime);

% ---- main function's end
end


% -------------------------------------------------------------------------
function  [net, outBlob] = process_runs(opts, numGpus, net, batchStruct, visitLayerID, outputBlobID, epitRange)
% -------------------------------------------------------------------------
% No Supoort for updating weights
    if isempty(batchStruct)
        return;
    end

    if numGpus >= 1
        one = gpuArray(single(1));
    else
        one = single(1);
    end

    if isfield(batchStruct, 'batchNumber') && strcmpi(opts.epit, 'epoch')
        globalBatchNum = batchStruct.batchNumber*(epitRange(1)-1)+1;
        [X,Y] = meshgrid(epitRange,1:batchStruct.batchNumber);
        epitRange = [reshape(X,1,[]); reshape(Y,1,[])]; % this creates a vector like: [1 1 1, 2 2 2, 3 3 3, ... ; 1 2 3 .., 1 2 3 .., 1 2 3 ..]
        rangeNumber = batchStruct.batchNumber;

    else
        rangeNumber = numel(epitRange);
        epitRange = [epitRange; 1:rangeNumber];
        globalBatchNum = epitRange(1);
    end

    mode = 'testing';
    opts.numToTest = 0;

    rangeNumber = batchStruct.batchNumber;
    epitRange = 1:batchStruct.N:batchStruct.m;
    epitRange = [ones(1,numel(epitRange));epitRange];
    globalBatchNum = epitRange(1);
    opts.displayIter = rangeNumber;

    %create options for forwardbackward
    optFB.accumulate = false;
    optFB.conserveMemory = opts.conserveMemory;
    optFB.sync = opts.sync;
    optFB.disableDropout = true;
    optFB.freezeDropout = false;
    optFB.visitLayerID = visitLayerID;
    optFB.outputBlobCount = cellfun(@numel, net.blobConnectId);
    optFB.gpuMode = numGpus >= 1;
    optFB.doder = false;

    res = [];
    count = 1;
    batchTime = tic ;
    outBlob = cell(1, numel(outputBlobID));
    for ob=1:numel(outputBlobID)
        blobsize = net.blobSizes{outputBlobID(ob)};
        outBlob{ob} = zeros(blobsize(1), blobsize(2), blobsize(3), batchStruct.m);
    end

    for t = epitRange
        % set learning rate
        learningRate = opts.learningRatePolicy(globalBatchNum, opts.learningRate, opts.learningRateGamma, opts.learningRatePower, opts.learningRateSteps) ;

        % get batch data
        [data, dataN, batchStruct] = nn.batch.fetch(batchStruct, opts.numSubBatches);
        if opts.prefetch
            [~, ~, batchStruct] = nn.batch.fetch(batchStruct, opts.numSubBatches);
        end

        %current subbatch number
        numSubBatches = numel(data);

        % run subbatches
        for s=1:numSubBatches
            % evaluate CNN
            dzdy = [];
            optFB.accumulate = s ~= 1;
            res = nn.forwardbackward(net, data{s}, dzdy, res, optFB);
        end

        lastIndices = batchStruct.lastBatchIndices(1, :);
        for ob = 1:numel(outBlob)
            tmp = gather(res.blob{outputBlobID(ob)});
            outBlob{ob}(:, :, :, lastIndices) = tmp;
        end

        count = count+1;
        globalBatchNum = globalBatchNum+1;

    end

end
