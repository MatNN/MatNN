function [net, batchStructTrain, batchStructVal] = train(net, batchStructTrain, batchStructVal, opts_user)
%TRAIN Partial code from VLfeat Library


opts.numEpochs = [] ;
opts.numInterations = [] ;
opts.numToSave = 10; %runs how many Epochs or iterations to save
opts.displayIter = 10; %Show info every opts.displayIter iterations
opts.batchSize = 256 ;
opts.numSubBatches = 1 ;
opts.gpus = [] ;

opts.learningRate = 0.001 ;
opts.learningRateGamma = 0.1;
opts.learningRatePower = 0.75;
opts.learningRateSteps = 1000;
opts.learningRatePolicy = @(currentBatchNumber, lr, gamma, power, steps) lr*(gamma^floor(currentBatchNumber/steps));
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;

opts.continue = [] ; % if you specify the saving's iteration/epoch number, then you can load it
opts.expDir = fullfile('data','exp') ;
opts.conserveMemory = false ;
opts.sync = false ;
opts.prefetch = false ;

opts.backPropDepth = +inf ;
opts.plotDiagnostics = false ;
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
opts = vl_argparse(opts, opts_user) ;

if isempty(batchStructTrain) && isempty(batchStructVal), error('Must specify Train/Val batch struct!!'); end
if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
    

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

evaluateMode = isempty(batchStructTrain) ;

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
if exist(opts.memoryMapFile), delete(opts.memoryMapFile) ; end



% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

% is epoch training or iterations
if strcmpi(opts.epit, 'epoch')
    runTimes = opts.numEpochs;
else
    runTimes = opts.numInterations;
end

%Load exist tmp savings
modelPath = @(ep) fullfile(opts.expDir, sprintf('%s-%s%d.mat', net.name, lower(opts.epit), ep));
if ~isempty(opts.continue)
    if exist(modelPath(opts.continue),'file')
        if opts.continue == runTimes
            fprintf('Load completed training network: %s\n', modelPath(opts.continue)) ;
        else
            fprintf('Resuming by loading %s%d\n', lower(opts.epit), opts.continue) ;
        end
        load(modelPath(opts.continue), 'net') ;
    else
        error('Can''t found savings.');
    end
end

startInd = 1;
if ~isempty(opts.continue)
    startInd = opts.continue+1;
end

saveTimes = numel(startInd:opts.numToSave:(runTimes-1))+1;
rngState = rng;
% start training from the last position
for i = startInd:opts.numToSave:(runTimes-1)
    % move CNN to GPU as needed
    if numGpus == 1
        net = nn.utils.movenet(net, 'gpu') ;
    elseif numGpus > 1
        spmd(numGpus)
            net_ = nn.utils.movenet(net, 'gpu') ;
        end
    end

    %iter/epoch range
    epitRange = i:min(i+opts.numToSave-1, runTimes);

    %Generate randomSeed use current rng settings
    randomSeed = randi(65536,1)-1;

    % train one epoch (or achieved opts.numToSave) and validate 
    % process_runs msut accept startIndex and endIndex of iter/epoch
    if numGpus <= 1
        %generate random seed
        rng(randomSeed);
        [net, batchStructTrain] = process_runs(true, opts, numGpus, net, batchStructTrain, epitRange) ;
        [net, batchStructVal] = process_runs(false, opts, numGpus, net, batchStructVal, epitRange) ;
    else
        spmd(numGpus)
            %generate random seed
            rng(randomSeed);
            [net_, batchStructTrain] = process_runs(true, opts, numGpus, net, batchStructTrain, epitRange) ;
            [net, batchStructVal] = process_runs(false, opts, numGpus, net, batchStructVal, epitRange) ;
        end
    end
    
    % save model file
    if numGpus > 1
        spmd(numGpus)
            net_ = nn.utils.movenet(net_, 'cpu') ;
        end
        net = net_{1} ;
    else
        net = nn.utils.movenet(net, 'cpu') ;
    end
    fprintf('Saving network model to %s ... \n',modelPath(epitRange(end)));
    if ~evaluateMode, save(modelPath(epitRange(end)), 'net'); end

end

%restores rng state
rng(rngState);


% ---- main function's end
end


% -------------------------------------------------------------------------
function  [net, batchStruct, prof] = process_runs(training, opts, numGpus, net, batchStruct, epitRange)
% -------------------------------------------------------------------------
    
    if isempty(batchStruct)
        return;
    end
    if training, mode = 'training' ; else, mode = 'validation' ; end
    if numlabs >1, mpiprofile on; end
    
    if numGpus >= 1
        one = gpuArray(single(1)) ;
    else
        one = single(1) ;
    end
    
    mmap = [] ;
    
    rangeNumber = 0;
    globalBatchNum  = 0;
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

    %find outputblobID
    outputBlobID = find(cellfun(@isempty, net.blobConnectId));
    accumulateOutBlobs = zeros(size(outputBlobID));
    res = [];
    count = 0;
    cumuTrainedDataNumber = 0;
    batchTime = tic ;
    for t = epitRange 
        % set learning rate
        learningRate = opts.learningRatePolicy(globalBatchNum, opts.learningRate, opts.learningRateGamma, opts.learningRatePower, opts.learningRateSteps) ;
        
        % get batch data
        [data, dataN, batchStruct] = nn.batch.fetch(batchStruct, numGpus >= 1, opts.numSubBatches);
        if opts.prefetch
            [~, ~, batchStruct] = nn.batch.fetch(batchStruct, numGpus >= 1, opts.numSubBatches);
        end

        %current subbatch number
        numSubBatches = numel(data);

        % run subbatches
        for s=1:numSubBatches
            % evaluate CNN
            if training, dzdy = one; else, dzdy = [] ; end
            res = nn.simplenn(net, data{s}, dzdy, res, ...
                         'accumulate', s ~= 1, ...
                         'disableDropout', ~training, ...
                         'conserveMemory', opts.conserveMemory, ...
                         'backPropDepth', opts.backPropDepth, ...
                         'sync', opts.sync, ...
                         'gpuMode', numGpus >= 1) ;
            % accumulate training errors
            % assume all output blobs is loss-like layer
            for ac = 1:numel(accumulateOutBlobs)
                blobRes = double(gather( res.blob{outputBlobID(ac)} ));
                accumulateOutBlobs(ac) = accumulateOutBlobs(ac) + sum(blobRes(:)) ;
            end
        end
        res.dzdwVisited = res.dzdwVisited & false;
        % gather and accumulate gradients across labs
        if training
            if numGpus <= 1
                net = accumulate_gradients(opts, learningRate, dataN, net, res) ;
            else
                if isempty(mmap)
                    mmap = map_gradients(opts.memoryMapFile, res, numGpus) ;
                end
                write_gradients(mmap, res) ;
                labBarrier() ;
                [net, res] = accumulate_gradients(opts, learningRate, dataN, net, res, mmap) ;
            end
        end

        % print learning statistics
        
        cumuTrainedDataNumber = cumuTrainedDataNumber+dataN;
        if mod(count, opts.displayIter) == 0
            fprintf('LabNo.%d - %s: %s %d (%d/%d), lr = %g ... ', labindex, mode, opts.epit, t(1), t(2), rangeNumber, learningRate) ; % eg. training iter 1600 (2/rangeNumber), lr = 0.001 ... %     training epoch 1 (2/batchNumber), lr = 0.001
            
            batchTime = toc(batchTime) ;
            speed = cumuTrainedDataNumber/batchTime ;
            if count == 0
                count = count +1;
            end
            for ac = 1:numel(accumulateOutBlobs)
                if isinf(accumulateOutBlobs(ac))
                    error('A blob output = Inf');
                end
                fprintf('blob(''%s'') = %.6g ', net.blobNames{outputBlobID(ac)}, accumulateOutBlobs(ac)/cumuTrainedDataNumber) ;
            end
            fprintf('%.2fs (%.1f data/s), ', batchTime, speed) ;
            fprintf('\n') ;
            
            batchTime = tic;
            accumulateOutBlobs = zeros(size(outputBlobID));
            cumuTrainedDataNumber = 0;
            count = 0;
        end
        
        % update batchStruct
        if numel(accumulateOutBlobs) == 1
            batchStruct.lastErrorRateOfData(batchStruct.lastBatchIndices) = accumulateOutBlobs(1)/cumuTrainedDataNumber;
        end

        count = count+1;
        globalBatchNum = globalBatchNum+1;

    end
    if numlabs >1
        prof = mpiprofile('info');
        mpiprofile off ;
    else
        prof = [];
    end
end






% -------------------------------------------------------------------------
function [net, res] = accumulate_gradients(opts, lr, batchSize, net, res, mmap)
% -------------------------------------------------------------------------

for w = 1:numel(res.dzdw)
    
    %debug info
    %fprintf('\nw=%d,%s=%.3f,%s=%.3f\n', w,'weightmax',max(net.weights{w}(:)), 'weightmin', min(net.weights{w}(:)));

    % accumualte from multiple labs (GPUs) if needed
    if nargin >= 6
      tag = sprintf('l%d',w) ;
      tmp = zeros(size(mmap.Data(labindex).(tag)), 'single') ;
      for g = setdiff(1:numel(mmap.Data), labindex)
        tmp = tmp + mmap.Data(g).(tag) ;
      end
      res.dzdw{w} = res.dzdw{w} + tmp ;
    end


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
    %        SOLVED!!!!, no need to no extra works. 以前想錯了。反正simplenn會幫你加起來
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
    net.momentum{w} = opts.momentum * net.momentum{w} - thisLR*(thisDecay * net.weights{w} + res.dzdw{w}) ;
    net.weights{w}  = net.weights{w} + net.momentum{w} ;
end


end
% -------------------------------------------------------------------------
function mmap = map_gradients(fname, res, numGpus)
% -------------------------------------------------------------------------
format = {} ;
for w=1:numel(res.dzdw)
    format(end+1,1:3) = {'single', size(res.dzdw{w}), sprintf('l%d',w)} ;
end
format(end+1,1:3) = {'double', [3 1], 'errors'} ;
if ~exist(fname) && (labindex == 1)
  f = fopen(fname,'wb') ;
  for g=1:numGpus
    for i=1:size(format,1)
      fwrite(f,zeros(format{i,2},format{i,1}),format{i,1}) ;
    end
  end
  fclose(f) ;
end
labBarrier() ;
mmap = memmapfile(fname, 'Format', format, 'Repeat', numGpus, 'Writable', true) ;
end
% -------------------------------------------------------------------------
function write_gradients(mmap, res)
% -------------------------------------------------------------------------
for w=1:numel(res.dzdw)
    mmap.Data(labindex).(sprintf('l%d',w)) = gather(res.dzdw{w}) ;
end
end
