function [net_trained] = mnist(baseNet)
if nargin == 1
    layers = nn.buildnet('MNIST', baseNet);
else
    layers = nn.buildnet('MNIST');
end

batchSize = 100;

layers.newLayer({
    'type' 'data.customData'...
    'name' 'data'...
    'top' {'data', 'label'} ...
    'customData_param' {
        'dataProvider' @MNISTDataProvider_train ...
        'batch_size'   batchSize ...
        'output_size'  {[28, 28, 1, batchSize], [1, 1, 1, batchSize]} ...
        'shuffle'      true ...
        }...
    'phase' 'train'
    });
layers.newLayer({
    'type' 'data.customData'...
    'name' 'data'...
    'top' {'data', 'label'} ...
    'customData_param' {
        'dataProvider' @MNISTDataProvider_test ...
        'batch_size'   100 ...
        'output_size'  {[28, 28, 1, 100], [1, 1, 1, 100]} ...
        'shuffle'      false ...
        }...
    'phase' 'test'
    });
layers.newLayer({
    'type'   'convolution'...
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
layers.newLayer({
    'type'   'pooling'...
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
layers.newLayer({
    'type'   'convolution'...
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
layers.newLayer({
    'type'   'pooling'...
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
layers.newLayer({
    'type'   'convolution'...
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
layers.newLayer({
    'type'   'relu'...
    'name'   'relu1'...
    'bottom' 'conv3'...
    'top'    'relu1'...
    });
layers.newLayer({
    'type'   'convolution'...
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
layers.newLayer({
    'type'   'loss.softmaxLoss'...
    'name'   'loss'...
    'bottom' {'fc4', 'label'}...
    'top'    'loss'...
    'phase'  'train'...
    });
layers.newLayer({
    'type'   'accuracy'...
    'name'   'accuracy'...
    'bottom' {'fc4', 'label'}...
    'top'    {'accuracy', 'meanClassAcc'}...
    'accuracy_param' {
        'meanClassAcc' true ...
        }...
    'phase'  'test'...
    });

[train4D, trainLabel, test4D, testLabel] = readMNISTDataset('mnist/train-images-idx3-ubyte', ...
                                                            'mnist/train-labels-idx1-ubyte', ...
                                                            'mnist/t10k-images-idx3-ubyte', ...
                                                            'mnist/t10k-labels-idx1-ubyte');

    pointer_train = 0;
    batchIndices_train = [];
    pointer_test = 0;
    batchIndices_test = [];

op.phaseOrder  = {'train'};
op.repeatTimes = 5;        % sum(opts.numToNext) * op.repeatTimes = total iterations
op.continue    = [];        % Set to <iter> to load specific intermediate model. eg. 300, 10, 36000
op.gpus        = [2];
op.expDir      = fullfile('data','exp');

trainOp.numToNext          = 600;  
trainOp.numToSave          = [];  
trainOp.displayIter        = 10;
trainOp.learningRateSteps  = 600;
trainOp.learningRatePolicy = @lrPolicy;

testOp.numToNext           = 100;
testOp.numToSave           = [];
testOp.displayIter         = 100;
testOp.showFirstIter       = false;
testOp.learningRate        = 0;

netObj = nn.nn(layers);
netObj.init(op);
netObj.initParameter(trainOp, 'train');
netObj.initParameter(testOp,  'test');
net_trained = netObj.run();



    
    function [top, weights, misc] = MNISTDataProvider_train(np, opts, l, weights, misc, bottom, top)
        bs = l.customData_param.batch_size;
        if numel(np.gpus)>1
            bs = bs/numel(np.gpus);
        end

        if pointer_train == 0
            batchIndices_train = zeros(1, ceil(numel(trainLabel)/bs)*bs);
            if l.customData_param.shuffle
                batchIndices_train(1:numel(trainLabel)) = randperm(numel(trainLabel));
            else
                batchIndices_train(1:numel(trainLabel)) = 1:numel(trainLabel);
            end
            batchIndices_train = reshape(batchIndices_train, bs, []);
            pointer_train = labindex;
        end

        % Get images
        imgInd = batchIndices_train(:, pointer_train);
        imgInd = imgInd(imgInd~=0);% last batch may have 0 indices
        if opts.gpuMode
            top{1} = gpuArray(single(train4D(:,:,1, imgInd)));
            top{2} = gpuArray(single(trainLabel(imgInd)));
        else
            top{1} = single(train4D(:,:,1, imgInd));
            top{2} = single(trainLabel(imgInd));
        end

        if pointer_train == size(batchIndices_train,2)
            pointer_train = 0;
        else
            pointer_train = mod(pointer_train+numel(np.gpus)  -1, size(batchIndices_train,2))  +1;
        end

    end


    
    function [top, weights, misc] = MNISTDataProvider_test(np, opts, l, weights, misc, bottom, top)
        bs = l.customData_param.batch_size;
        if numel(np.gpus)>1
            bs = bs/numel(np.gpus);
        end

        if pointer_test == 0
            batchIndices_test = zeros(1, ceil(numel(testLabel)/bs)*bs);
            if l.customData_param.shuffle
                batchIndices_test(1:numel(testLabel)) = randperm(numel(testLabel));
            else
                batchIndices_test(1:numel(testLabel)) = 1:numel(testLabel);
            end
            batchIndices_test = reshape(batchIndices_test, bs, []);
            pointer_test = 1;
        end

        % Get images
        imgInd = batchIndices_test(:, pointer_test);
        imgInd = imgInd(imgInd~=0);% last batch may have 0 indices
        if opts.gpuMode
            top{1} = gpuArray(single(test4D(:,:,1, imgInd)));
            top{2} = gpuArray(single(reshape(testLabel(imgInd),1,1,1,[])));
        else
            top{1} = single(test4D(:,:,1, imgInd));
            top{2} = single(reshape(testLabel(imgInd),1,1,1,[]));
        end


        if pointer_test == size(batchIndices_test,2)
            pointer_test = 0;
        else
            pointer_test = pointer_test+1;
        end

    end



end

function res = lrPolicy(globalIterNum, currentPhaseTotalIter, lr, gamma, power, steps)
     res = lr*(gamma^floor((currentPhaseTotalIter-1)/steps));
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