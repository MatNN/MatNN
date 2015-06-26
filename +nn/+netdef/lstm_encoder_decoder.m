function [net_trained] = lstm_encoder_decoder(baseNet)
if nargin == 1
    no = nn.buildnet('LSTM', baseNet);
else
    no = nn.buildnet('LSTM');
end

batchSize = 64;
labelBlobName_t1 = 'target_t1';
labelBlobName_t2 = 'target_t2';
labelBlobName_t3 = 'target_t3';
labelBlobName_t4 = 'target_t4';
dataBlobName_t1  = 'data_t1';
dataBlobName_t2  = 'data_t2';
dataBlobName_t3  = 'data_t3';
dataBlobName_t4  = 'data_t4';

% set the data-format same as training the network, no need to separate the data
% into training and testing set.
no.setDataBlobSize(dataBlobName_t1 , [1 1 128 batchSize]);
no.setDataBlobSize(dataBlobName_t2 , [1 1 128 batchSize]);
no.setDataBlobSize(dataBlobName_t3 , [1 1 128 batchSize]);
no.setDataBlobSize(dataBlobName_t4 , [1 1 128 batchSize]);
no.setDataBlobSize(labelBlobName_t1, [1 1 128 batchSize]);
no.setDataBlobSize(labelBlobName_t2, [1 1 128 batchSize]);
no.setDataBlobSize(labelBlobName_t3, [1 1 128 batchSize]);
no.setDataBlobSize(labelBlobName_t4, [1 1 128 batchSize]);

%%%%%%%%% define model of encoder LSTM %%%%%%%%%
no = nn.netdef.model_lstm_encoder(no, 1);
no = nn.netdef.model_lstm_encoder(no, 2);
no = nn.netdef.model_lstm_encoder(no, 3);
no = nn.netdef.model_lstm_encoder(no, 4);

no.newLayer({
        'type'   'layers.convolution'...
        'name'   'codex'...
        'bottom' 'encoder_final_t4' ...
        'top'    'codex'...
        'convolution_param' {
            'num_output'  128 ...
            'kernel_size' 1  ...
            'pad'         0  ...
            'stride'      1  ...
            }...
        'weight_param' {
            'name'         {'codex_w', 'codex_b'} ...
            'generator'    {@nn.generator.xavier, @nn.generator.constant} ...
            'learningRate' [1 2]
            }
        });

no = nn.netdef.model_lstm_decoder(no, 1);
no = nn.netdef.model_lstm_decoder(no, 2);
no = nn.netdef.model_lstm_decoder(no, 3);
no = nn.netdef.model_lstm_decoder(no, 4);

%%%%%%%%% loss %%%%%%%%%
no.newLayer({
    'type'   'layers.loss.euclideanLoss'...
    'name'   'loss_t1'...
    'bottom' {'decoder_final_t1', 'target_t1'}...
    'top'    'loss_t1' ...
    });
no.newLayer({
    'type'   'layers.loss.euclideanLoss'...
    'name'   'loss_t2'...
    'bottom' {'decoder_final_t2', 'target_t2'}...
    'top'    'loss_t2' ...
    });
no.newLayer({
    'type'   'layers.loss.euclideanLoss'...
    'name'   'loss_t3'...
    'bottom' {'decoder_final_t3', 'target_t3'}...
    'top'    'loss_t3' ...
    });
no.newLayer({
    'type'   'layers.loss.euclideanLoss'...
    'name'   'loss_t4'...
    'bottom' {'decoder_final_t4', 'target_t4'}...
    'top'    'loss_t4' ...
    });

trainObj = readCifarDataset('cifarData.mat');

dataStruct_t1  = nn.batch.generate(false, 'Name', dataBlobName_t1,  'File', trainObj.train4D_t1, 'BatchSize', batchSize);
dataStruct_t2 = nn.batch.generate(false, 'Name', dataBlobName_t2, 'File', trainObj.train4D_t2, 'BatchSize', batchSize);
dataStruct_t3  = nn.batch.generate(false, 'Name', dataBlobName_t3,  'File', trainObj.train4D_t3, 'BatchSize', batchSize);
dataStruct_t4 = nn.batch.generate(false, 'Name', dataBlobName_t4, 'File', trainObj.train4D_t4, 'BatchSize', batchSize);
labelStruct_t1 = nn.batch.generate(false, 'Name', labelBlobName_t1, 'File', trainObj.trainLabel_t1, 'BatchSize', batchSize);
labelStruct_t2 = nn.batch.generate(false, 'Name', labelBlobName_t2, 'File', trainObj.trainLabel_t2, 'BatchSize', batchSize);
labelStruct_t3 = nn.batch.generate(false, 'Name', labelBlobName_t3, 'File', trainObj.trainLabel_t3, 'BatchSize', batchSize);
labelStruct_t4 = nn.batch.generate(false, 'Name', labelBlobName_t4, 'File', trainObj.trainLabel_t4, 'BatchSize', batchSize);
batchStruct = nn.batch.generate('Attach', dataStruct_t1, dataStruct_t2, dataStruct_t3, dataStruct_t4, labelStruct_t1, labelStruct_t2, labelStruct_t3, labelStruct_t4);

opts.numEpochs = 1 ;
opts.numInterations = [] ;
opts.numToSave = 5000; %runs how many Epochs or iterations to save
opts.numToTest = [];
opts.displayIter = 100;
opts.batchSize = batchSize ;
opts.numSubBatches = 1 ;
opts.gpus = [1];
opts.computeMode = 'cuda kernel';

opts.learningRate = 0.0001;
opts.learningRatePolicy = @lrPolicy; %every iteration decays the lr
opts.learningRateGamma = 0.0001;
opts.learningRatePower = 0.75;
opts.weightDecay = 0.0002;

opts.continue = []; % if you specify the saving's iteration/epoch number, you can load it
opts.expDir = fullfile('~/lstm/lstm_encoder_decoder','exp') ;
opts.conserveMemory = false ;
opts.sync = false ;
opts.prefetch = false ;

[net_trained, batchStructTrained, ~] = nn.train(no, batchStruct, [], opts);

end

function res = lrPolicy(currentBatchNumber, lr, gamma, power, steps)
     res = lr*((1+gamma*currentBatchNumber)^(-power));
end

function [trainObj, testObj] = readCifarDataset(file, seed)
% Random permute the training data with random seed.
    load(file);
    clearvars trainLabels testLabels;
    meanData = mean([trainData; testData]);
    trainData = bsxfun(@minus, trainData, meanData);
    testData = bsxfun(@minus, testData, meanData);
    [~, ~, w_pca] = pcasvd(trainData, 512);

    nTrain = numel(trainData)/512;
    trainObj.trainLabel_t1 = reshape(w_pca(:, 1:128)'*trainData', [1 1 128 nTrain]);
    trainObj.trainLabel_t2 = reshape(w_pca(:, 129:256)'*trainData', [1 1 128 nTrain]);
    trainObj.trainLabel_t3 = reshape(w_pca(:, 257:384)'*trainData', [1 1 128 nTrain]);
    trainObj.trainLabel_t4 = reshape(w_pca(:, 385:512)'*trainData', [1 1 128 nTrain]);

    nTest = numel(testData)/512;
    testObj.testLabel_t1 = reshape(w_pca(:, 1:128)'*testData', [1 1 128 nTest]);
    testObj.testLabel_t2 = reshape(w_pca(:, 129:256)'*testData', [1 1 128 nTest]);
    testObj.testLabel_t3 = reshape(w_pca(:, 257:384)'*testData', [1 1 128 nTest]);
    testObj.testLabel_t4 = reshape(w_pca(:, 385:512)'*testData', [1 1 128 nTest]);

    for nn = 1:nTrain
        order = (randperm(4,4)-1)*128;
        trainObj.train4D_t1(:, :, :, nn) = reshape(trainData(nn, order(1)+1:order(1)+128)', [1 1 128 1]);
        trainObj.train4D_t2(:, :, :, nn) = reshape(trainData(nn, order(2)+1:order(2)+128)', [1 1 128 1]);
        trainObj.train4D_t3(:, :, :, nn) = reshape(trainData(nn, order(3)+1:order(3)+128)', [1 1 128 1]);
        trainObj.train4D_t4(:, :, :, nn) = reshape(trainData(nn, order(4)+1:order(4)+128)', [1 1 128 1]);
    end
    for nn = 1:nTest
        order = (randperm(4,4)-1)*128;
        testObj.test4D_t1(:, :, :, nn) = reshape(testData(nn, order(1)+1:order(1)+128)', [1 1 128 1]);
        testObj.test4D_t2(:, :, :, nn) = reshape(testData(nn, order(2)+1:order(2)+128)', [1 1 128 1]);
        testObj.test4D_t3(:, :, :, nn) = reshape(testData(nn, order(3)+1:order(3)+128)', [1 1 128 1]);
        testObj.test4D_t4(:, :, :, nn) = reshape(testData(nn, order(4)+1:order(4)+128)', [1 1 128 1]);
    end
end