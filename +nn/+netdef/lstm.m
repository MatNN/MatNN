function [net_trained] = lstm(baseNet)
if nargin == 1
    no = nn.buildnet('LSTM', baseNet);
else
    no = nn.buildnet('LSTM');
end

batchSize = 64;
labelBlobName_t1 = 'target_t1';
labelBlobName_t2 = 'target_t2';
dataBlobName_t1  = 'data_t1';
dataBlobName_t2  = 'data_t2';

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

[trainObj, testObj] = readCifarDataset('cifarData');

dataStruct_t1  = nn.batch.generate(false, 'Name', dataBlobName_t1,  'File', trainObj.train4D_t1, 'BatchSize', batchSize);
dataStruct_t2 = nn.batch.generate(false, 'Name', dataBlobName_t2, 'File', trainObj.train4D_t2, 'BatchSize', batchSize);
labelStruct_t1 = nn.batch.generate(false, 'Name', labelBlobName_t1, 'File', trainObj.trainLabel_t1, 'BatchSize', batchSize);
labelStruct_t2 = nn.batch.generate(false, 'Name', labelBlobName_t2, 'File', trainObj.trainLabel_t2, 'BatchSize', batchSize);
batchStruct = nn.batch.generate('Attach', dataStruct_t1, dataStruct_t2, labelStruct_t1, labelStruct_t2);

opts.numEpochs = 1 ;
opts.numInterations = [] ;
opts.numToSave = 5000; %runs how many Epochs or iterations to save
opts.numToTest = [];
opts.displayIter = 100;
opts.batchSize = batchSize ;
opts.numSubBatches = 1 ;
opts.gpus = [1];
opts.computeMode = 'cuda kernel';

opts.learningRate = 0.03 ;
opts.learningRatePolicy = @lrPolicy; %every iteration decays the lr
opts.learningRateGamma = 0.0001;
opts.learningRatePower = 0.75;
opts.weightDecay = 0.0002;

opts.continue = []; % if you specify the saving's iteration/epoch number, you can load it
opts.expDir = fullfile('data','exp') ;
opts.conserveMemory = false ;
opts.sync = false ;
opts.prefetch = false ;

[net_trained, batchStructTrained, ~] = nn.train(no, batchStruct, [], opts);

end

function res = lrPolicy(currentBatchNumber, lr, gamma, power, steps)
     res = lr*((1+gamma*currentBatchNumber)^(-power));
end

function [trainObj, testObj] = readCifarDataset(file)

    load(file);
    clearvars trainLabels testLabels;
    meanData = mean([trainData; testData]);
    trainData = bsxfun(@minus, trainData, meanData);
    testData = bsxfun(@minus, testData, meanData);
    [~, ~, w_t1] = pcasvd(trainData(:, 1:256), 256);
    [~, ~, w_t2] = pcasvd(trainData, 256);

    nTrain = numel(trainData)/512;
    trainObj.train4D_t1 = reshape(trainData(:, 1:256)', [1 1 256 nTrain]);
    trainObj.train4D_t2 = reshape(trainData(:, 257:512)', [1 1 256 nTrain]);
    trainObj.trainLabel_t1 = reshape(w_t1'*trainData(:, 1:256)', [1 1 256 nTrain]);
    trainObj.trainLabel_t2 = reshape(w_t2'*trainData', [1 1 256 nTrain]);

    nTest = numel(testData)/512;
    testObj.test4D_t1 = reshape(testData(:, 1:256)', [1 1 256 nTest]);
    testObj.test4D_t2 = reshape(testData(:, 257:512)', [1 1 256 nTest]);
    testObj.testLabel_t1 = reshape(w_t1'*testData(:, 1:256)', [1 1 256 nTest]);
    testObj.testLabel_t2 = reshape(w_t2'*testData', [1 1 256 nTest]);
end