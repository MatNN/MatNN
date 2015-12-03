function mnist_spatialTransform(varargin)
if numel(varargin) == 0
    showWindow = true;
else
    showWindow = varargin{1};
end

conf = nn.examples.config();
trainer = nn.nn('MNIST_SpatialTransform');
batchSize = 100;

trainer.add({
    'type' 'data.MNIST'...
    'name' 'data_train'...
    'top' {'data', 'label'} ...
    'data_param' {
                'src' conf.mnistPath ...
        'root_folder' ''       ...
         'batch_size' batchSize ...
               'full' false  ...
            'shuffle' true   ...
        } ...
    'mnist_param' {
           'type' 'train' ...
        } ...
    'phase' 'train'
    });
trainer.add({
    'type' 'data.MNIST'...
    'name' 'data_test'...
    'top' {'data', 'label'} ...
    'data_param' {
                'src' conf.mnistPath ...
        'root_folder' ''       ...
         'batch_size' batchSize ...
               'full' false  ...
            'shuffle' true   ...
        } ...
    'mnist_param' {
           'type' 'test' ...
        } ...
    'phase' 'test'
    });
%------------------Spatial Transform
trainer.add({
    'type'   'transform.RandDistort' ...
    'name'   'RandomDistortion' ...
    'bottom' 'data' ...
    'top'    'data2' ...
    'randDistort_param' {
        'angle'  [-90,  90] ...
        'scaleX' [1,  1.42] ...
        'scaleY' [1,  1.42] ...
        'scaleEQ' true ...
        'shiftX' [-0.7,  0.7] ...
        'shiftY' [-0.7,  0.7] ...
        'extend' [4,  4] ...
        'mix'    2 ...
        }
    })
%'scaleX' [0.83,  1.42] ...
trainer.add({
    'type'   'Pooling'...
    'name'   'lp1'...
    'bottom' 'data2'...
    'top'    'lp1'...
    'pooling_param' {
        'method'      'max' ...
        'kernel_size' 2  ...
        'pad'         0  ...
        'stride'      2  ...
        }...
    });
trainer.add({
    'type'   'Conv'...
    'name'   'local1'...
    'bottom' 'lp1' ...
    'top'    'local1'...
    'conv_param' {
        'num_output'  20 ...
        'kernel_size' 5  ...
        'pad'         0  ...
        'stride'      1  ...
        }...
    'weight_param' {
        'generator'   {@nn.generator.gaussian, @nn.generator.constant} ...
        'generator_param' {{'mean', 0, 'std', 0.01}, []} ...
        'learningRate' single([1,1]) ...
        }
    });
trainer.add({
   'type'   'ReLU'...
   'name'   'lr1'...
   'bottom' 'local1'...
   'top'    'local1'...
   });
trainer.add({
   'type'   'Pooling'...
   'name'   'lp2'...
   'bottom' 'local1'...
   'top'    'lp2'...
   'pooling_param' {
       'method'      'max' ...
       'kernel_size' 2  ...
       'pad'         0  ...
       'stride'      2  ...
       }...
   });
trainer.add({
    'type'   'Conv'...
    'name'   'local2'...
    'bottom' 'lp2' ...
    'top'    'local2'...
    'conv_param' {
        'num_output'  20 ...
        'kernel_size' 5  ...
        'pad'         0  ...
        'stride'      1  ...
        }...
    'weight_param' {
        'generator'   {@nn.generator.gaussian, @nn.generator.constant} ...
        'generator_param' {{'mean', 0, 'std', 0.01}, []} ...
        'learningRate' single([1,1]) ...
        }
    });
trainer.add({
    'type'   'ReLU'...
    'name'   'lr2'...
    'bottom' 'local2'...
    'top'    'local2'...
});
trainer.add({
    'type'   'Conv'...
    'name'   'local3'...
    'bottom' 'local2' ...
    'top'    'local3'...
    'conv_param' {
        'num_output'  20 ...
        'kernel_size' 2  ...
        'pad'         0  ...
        'stride'      1  ...
        }...
    'weight_param' {
        'generator'   {@nn.generator.gaussian, @nn.generator.constant} ...
        'generator_param' {{'mean', 0, 'std', 0.01}, []} ...
        'learningRate' single([1,1]) ...
        }
    });
trainer.add({
    'type'   'ReLU'...
    'name'   'lr3'...
    'bottom' 'local3'...
    'top'    'local3'...
    });
trainer.add({
    'type'   'Conv'...
    'name'   'local4'...
    'bottom' 'local3' ...
    'top'    'local4'...
    'conv_param' {
        'num_output'  6  ...
        'kernel_size' 1  ...
        'pad'         0  ...
        'stride'      1  ...
        }...
    'weight_param' {
        'generator'   {@nn.generator.constant, @(d,~) reshape(single([0.3, 0, 0, 0.3, 0, 0]),d)} ...
        'generator_param' {{}, {}} ...
        'learningRate' single([1,1]) ...
        }
    });

trainer.add({
    'type'   'transform.Affine' ...
    'name'   'localTransform' ...
    'bottom' {'data2', 'local4'} ...
    'top'    'lt' ...
    'affine_param' {...
        'showDebugWindow' showWindow ...
        }
    });
%------------------End of Spatial Transform

trainer.add({
    'type'   'Conv'...
    'name'   'conv1'...
    'bottom' 'lt' ...
    'top'    'conv1'...
    'conv_param' {
        'num_output'  128 ...
        'kernel_size' 32  ...
        'pad'         0  ...
        'stride'      1  ...
        }...
    'weight_param' {
        'generator'   {@nn.generator.gaussian, @nn.generator.constant} ...
        'generator_param' {{'mean', 0, 'std', 0.01}, []}
        }
    });
trainer.add({
    'type'   'ReLU'...
    'name'   'relu1'...
    'bottom' 'conv1'...
    'top'    'conv1'...
    });
trainer.add({
    'type'   'Conv'...
    'name'   'conv2'...
    'bottom' 'conv1' ...
    'top'    'conv2'...
    'conv_param' {
        'num_output'  128 ...
        'kernel_size' 1  ...
        'pad'         0  ...
        'stride'      1  ...
        }...
    'weight_param' {
        'generator'   {@nn.generator.gaussian, @nn.generator.constant} ...
        'generator_param' {{'mean', 0, 'std', 0.01}, []}
        }
    });
trainer.add({
    'type'   'ReLU'...
    'name'   'relu2'...
    'bottom' 'conv2'...
    'top'    'conv2'...
    });
trainer.add({
    'type'   'Conv'...
    'name'   'conv3'...
    'bottom' 'conv2' ...
    'top'    'conv3'...
    'conv_param' {
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
trainer.add({
    'type'   'loss.SoftMaxLoss'...
    'name'   'loss'...
    'bottom' {'conv3', 'label'}...
    'top'    'loss2'...
    'phase'  'train'...
    });
trainer.add({
    'type'   'Accuracy'...
    'name'   'accuracy'...
    'bottom' {'conv3', 'label'}...
    'top'    {'accuracy', 'meanClassAcc'}...
    'accuracy_param' {
        'meanClassAcc' true ...
        }...
    'phase'  'test'...
    });


trainer.setPhaseOrder('train', 'test');
trainer.setRepeat(100);
trainer.setSavePath(fullfile('data','exp'));
trainer.setGpu(1);

trainOp.numToNext          = 600;  
trainOp.numToSave          = 600*conf.save;  
trainOp.displayIter        = 20;
trainOp.learningRateSteps  = 600;
trainOp.learningRateGamma  = 0.95;
trainOp.learningRatePolicy = @lrPolicy;
trainer.setPhasePara('train', trainOp);

testOp.numToNext           = 100;
testOp.numToSave           = [];
testOp.displayIter         = 100;
testOp.showFirstIter       = false;
testOp.learningRate        = 0;
trainer.setPhasePara('test', testOp);

trainer.run();

end

function res = lrPolicy(globalIterNum, currentPhaseTotalIter, lr, gamma, power, steps)
     %res = lr*(gamma^floor((currentPhaseTotalIter-1)/steps));
     res = lr;
end