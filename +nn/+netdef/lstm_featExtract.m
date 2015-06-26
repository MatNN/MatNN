function [outBlob] = lstm_featExtract(baseNet, blobNames)

assert(~isempty(baseNet));
assert(~isempty(blobNames));

no = nn.buildnet('LSTM', baseNet);

batchSize = 64;
labelBlobName_t1 = 'target_t1';
labelBlobName_t2 = 'target_t2';
dataBlobName_t1  = 'data_t1';
dataBlobName_t2  = 'data_t2';

% set the data-format same as training the network, no need to separate the data
% into training and testing set.
no.setDataBlobSize(dataBlobName_t1 , [1 1 256 batchSize]);
no.setDataBlobSize(dataBlobName_t2 , [1 1 256 batchSize]);
no.setDataBlobSize(labelBlobName_t1, [1 1 256 batchSize]);
no.setDataBlobSize(labelBlobName_t2, [1 1 256 batchSize]);

%%%%%%%%% define model of encoder LSTM %%%%%%%%%
no = nn.netdef.model_lstm(no, 1);
no = nn.netdef.model_lstm(no, 2);

%%%%%%%%% loss %%%%%%%%%
no.newLayer({
    'type'   'layers.loss.euclideanLoss'...
    'name'   'loss_t1'...
    'bottom' {'final_t1', 'target_t1'}...
    'top'    'loss_t1' ...
    'phase' 'train'
    });
no.newLayer({
    'type'   'layers.loss.euclideanLoss'...
    'name'   'loss_t2'...
    'bottom' {'final_t2', 'target_t2'}...
    'top'    'loss_t2' ...
    'phase' 'train'
    });


trainObj = readCifarDataset('cifarData');

dataStruct_t1  = nn.batch.generate(false, 'Name', dataBlobName_t1,  'File', trainObj.train4D_t1, 'BatchSize', batchSize);
dataStruct_t2 = nn.batch.generate(false, 'Name', dataBlobName_t2, 'File', trainObj.train4D_t2, 'BatchSize', batchSize);
labelStruct_t1 = nn.batch.generate(false, 'Name', labelBlobName_t1, 'File', trainObj.trainLabel_t1, 'BatchSize', batchSize);
labelStruct_t2 = nn.batch.generate(false, 'Name', labelBlobName_t2, 'File', trainObj.trainLabel_t2, 'BatchSize', batchSize);
batchStruct = nn.batch.generate('Attach', dataStruct_t1, dataStruct_t2, labelStruct_t1, labelStruct_t2);
opts.numEpochs = [] ;
opts.numInterations = 1;
opts.numToSave = []; %runs how many Epochs or iterations to save
opts.numToTest = [];
opts.displayIter = [];
opts.batchSize = batchSize ;
opts.numSubBatches = 1 ;
opts.gpus = [1];
opts.computeMode = 'cuda kernel';

opts.continue = []; % if you specify the saving's iteration/epoch number, you can load it
opts.expDir = fullfile('~/lstm/lstm','exp') ;
opts.conserveMemory = false ;
opts.sync = false ;
opts.prefetch = false ;

[net_trained, outBlob] = nn.featextract(no, blobNames, batchStruct, opts);

end

function [trainObj] = readCifarDataset(file)

    load(file);
    clearvars trainLabels testLabels;
    trainData = [trainData; testData];
    trainData = bsxfun(@minus, trainData, mean(trainData));
    [~, ~, w_t1] = pcasvd(trainData(:, 1:256), 256);
    [~, ~, w_t2] = pcasvd(trainData, 256);

    nTrain = numel(trainData)/512;
    trainObj.train4D_t1 = reshape(trainData(:, 1:256)', [1 1 256 nTrain]);
    trainObj.train4D_t2 = reshape(trainData(:, 257:512)', [1 1 256 nTrain]);
    trainObj.trainLabel_t1 = reshape(w_t1'*trainData(:, 1:256)', [1 1 256 nTrain]);
    trainObj.trainLabel_t2 = reshape(w_t2'*trainData', [1 1 256 nTrain]);

end