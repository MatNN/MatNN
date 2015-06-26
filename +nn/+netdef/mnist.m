function [net_trained] = mnist(baseNet)
if nargin == 1
    no = nn.buildnet('MNIST', baseNet);
else
    no = nn.buildnet('MNIST');
end

batchSize = 100;
labelBlobName = 'label';
dataBlobName  = 'data';

no.setDataBlobSize(dataBlobName , [28 28 1 batchSize]);
no.setDataBlobSize(labelBlobName, [1  1  1 batchSize]);

no.newLayer({
    'type'   'layers.convolution'...
    'name'   'conv1'...
    'bottom' 'data' ...
    'top'    'conv1'...
    'convolution_param' {
        'num_output'  20 ...
        'kernel_size' 5  ...
        'pad'         0  ...
        'stride'      1  ...
        }...
    'weight_param' {
        'generator'   {@nn.generator.gaussian, @nn.generator.constant} ...
        'generator_param' {{'mean', 0, 'std', 0.01}, []}
        }
    });
no.newLayer({
    'type'   'layers.pooling'...
    'name'   'pool1'...
    'bottom' 'conv1'...
    'top'    'pool1'...
    'pooling_param' {
        'method'      'max' ...
        'kernel_size' 2  ...
        'pad'         0  ...
        'stride'      2  ...
        }...
    });
no.newLayer({
    'type'   'layers.convolution'...
    'name'   'conv2'...
    'bottom' 'pool1' ...
    'top'    'conv2'...
    'convolution_param' {
        'num_output'  50 ...
        'kernel_size' 5  ...
        'pad'         0  ...
        'stride'      1  ...
        }...
    'weight_param' {
        'generator'   {@nn.generator.gaussian, @nn.generator.constant} ...
        'generator_param' {{'mean', 0, 'std', 0.01}, []}
        }
    });
no.newLayer({
    'type'   'layers.pooling'...
    'name'   'pool2'...
    'bottom' 'conv2'...
    'top'    'pool2'...
    'pooling_param' {
        'method'      'max' ...
        'kernel_size' 2  ...
        'pad'         0  ...
        'stride'      2  ...
        }...
    });
no.newLayer({
    'type'   'layers.convolution'...
    'name'   'conv3'...
    'bottom' 'pool2' ...
    'top'    'conv3'...
    'convolution_param' {
        'num_output'  500 ...
        'kernel_size' 4  ...
        'pad'         0  ...
        'stride'      1  ...
        }...
    'weight_param' {
        'generator'   {@nn.generator.gaussian, @nn.generator.constant} ...
        'generator_param' {{'mean', 0, 'std', 0.01}, []}
        }
    });
no.newLayer({
    'type'   'layers.relu'...
    'name'   'relu1'...
    'bottom' 'conv3'...
    'top'    'relu1'...
    });
no.newLayer({
    'type'   'layers.convolution'...
    'name'   'fc4'...
    'bottom' 'relu1' ...
    'top'    'fc4'...
    'convolution_param' {
        'num_output'  10 ...
        'kernel_size' 1  ...
        'pad'         0  ...
        'stride'      1  ...
        }...
    'weight_param' {
        'generator'   {@nn.generator.gaussian, @nn.generator.constant} ...
        'generator_param' {{'mean', 0, 'std', 0.01}, []}
        }
    });
no.newLayer({
    'type'   'layers.loss.softmaxLoss'...
    'name'   'loss'...
    'bottom' {'fc4', 'label'}...
    'top'    'loss'...
    'phase'  'train'...
    });
no.newLayer({
    'type'   'layers.accuracy'...
    'name'   'accuracy'...
    'bottom' {'fc4', 'label'}...
    'top'    'accuracy'...
    'phase'  'test'...
    });

[train4D, trainLabel, test4D, testLabel] = readMNISTDataset('train-images-idx3-ubyte', ...
                                                            'train-labels-idx1-ubyte', ...
                                                            't10k-images-idx3-ubyte', ...
                                                            't10k-labels-idx1-ubyte');


dataStruct  = nn.batch.generate(false, 'Name', dataBlobName,  'File', train4D,    'BatchSize', batchSize, 'Random', 2);
labelStruct = nn.batch.generate(false, 'Name', labelBlobName, 'File', trainLabel', 'BatchSize', batchSize, 'Random', 2, 'Using4D', false);
batchStruct = nn.batch.generate('Attach', dataStruct, labelStruct);

vDataStruct  = nn.batch.generate(false, 'Name', dataBlobName,  'File', test4D,    'BatchSize', batchSize, 'Random', 0);
vLabelStruct = nn.batch.generate(false, 'Name', labelBlobName, 'File', testLabel', 'BatchSize', batchSize, 'Random', 0, 'Using4D', false);
vBatchStruct = nn.batch.generate('Attach', vDataStruct, vLabelStruct);

opts.numEpochs = [] ;
opts.numInterations = 12000 ;
opts.numToTest = 300;
opts.numToSave = 600; %runs how many Epochs or iterations to save
opts.displayIter = 60;
opts.batchSize = batchSize ;
opts.numSubBatches = 1 ;
opts.gpus = [] ; %#ok
opts.computeMode = 'default';

opts.learningRate = 0.001 ;
opts.learningRatePolicy = @lrPolicy; %every iteration decays the lr
opts.learningRateGamma = 0.0001;
opts.learningRatePower = 0.75;

opts.continue = [] ; % if you specify the saving's iteration/epoch number, then you can load it
opts.expDir = fullfile('data','exp') ;
opts.conserveMemory = false ;
opts.sync = false ;
opts.prefetch = false ;


[net_trained, batchStructTrained, ~] = nn.train(no, batchStruct, vBatchStruct, opts);

end

function res = lrPolicy(currentBatchNumber, lr, gamma, power, steps)
     res = lr*((1+gamma*currentBatchNumber)^(-power));
end

function [train4D, trainLabel, test4D, testLabel] = readMNISTDataset(trainImgFile, trainLabelFile, testImgFile, testLabelFile)
    m = memmapfile(trainImgFile,'Offset', 16,'Format', {'uint8' [28 28] 'img'});
    imgData = m.Data;
    clearvars m;
    train4D = zeros(28,28,1,numel(imgData), 'uint8');
    for i=1:numel(imgData)
        train4D(:,:,1,i) = imgData(i).img';
    end

    m = memmapfile(testImgFile,'Offset', 16,'Format', {'uint8' [28 28] 'img'});
    imgData = m.Data;
    clearvars m;
    test4D = zeros(28,28,1,numel(imgData), 'uint8');
    for i=1:numel(imgData)
        test4D(:,:,1,i) = imgData(i).img';
    end

    m = memmapfile(trainLabelFile,'Offset', 8,'Format', 'uint8');
    trainLabel = m.Data;
    m = memmapfile(testLabelFile, 'Offset', 8,'Format', 'uint8');
    testLabel = m.Data;
    clearvars m;

end
