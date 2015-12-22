function mnist_bipo(varargin)
%MNIST_BIPO bilinear interpolation

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
%------------------Spatial Transform
trainer.add({
    'type'   'transform.RandDistort' ...
    'name'   'RandomDistortion' ...
    'bottom' 'data' ...
    'top'    'data2' ...
    'randDistort_param' {
        'angle'  [-90,  90] ...
        'scaleX' [0.83,  1.42] ...
        'scaleY' [0.83,  1.42] ...
        'scaleEQ' true ...
        'shiftX' [-0.5,  0.5] ...
        'shiftY' [-0.5,  0.5] ...
        'extend' [4,  4] ...
        'mix'    0 ...
        }
    })
trainer.add({
    'type'   'Conv'...
    'name'   'local1'...
    'bottom' 'data2' ...
    'top'    'local1'...
    'conv_param' {
        'num_output'  4096  ...
        'kernel_size' 32 ...
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
    'type'   'Conv'...
    'name'   'local2'...
    'bottom' 'local1' ...
    'top'    'local2'...
    'conv_param' {
        'num_output'  4096  ...
        'kernel_size' 1  ...
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
        'num_output'  4096  ...
        'kernel_size' 1  ...
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
        'num_output'  4096  ...
        'kernel_size' 1  ...
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
    'name'   'lr4'...
    'bottom' 'local4'...
    'top'    'local4'...
    });
trainer.add({
    'type'   'Conv'...
    'name'   'local5'...
    'bottom' 'local4' ...
    'top'    'local5'...
    'conv_param' {
        'num_output'  2048  ...
        'kernel_size' 1  ...
        'pad'         0  ...
        'stride'      1  ...
        }...
    'weight_param' {
        'generator'   {@nn.generator.constant, @biInit} ...
        'generator_param' {{}, {}} ...
        'learningRate' single([1,1]) ...
        }
    });
trainer.add({
    'type'   'Reshape'...
    'name'   'reshape'...
    'bottom' 'local5' ...
    'top'    'local5R'...
    'reshape_param' {
        'output_size' {32,32,2,[]}
        }
    });

trainer.add({
    'type'   'transform.BilinearInterpolation' ...
    'name'   'localTransform' ...
    'bottom' {'data2', 'local5R'} ...
    'top'    'lt' ...
    'bilinear_param' {...
        'showDebugWindow' showWindow ...
        }
    });
%------------------End of Spatial Transform
trainer.add({
    'type' 'Crop' ...
    'name' 'crop' ...
    'bottom' {'lt', 'data'} ...
    'top' 'lt_crop'
    });

trainer.add({
    'type'   'loss.EuclideanLoss'...
    'name'   'loss'...
    'bottom' {'lt_crop', 'data'}...
    'top'    'loss2'...
    'loss_param' { ...
        'loss_weight' 0.0005 ...
    } ...
    'phase'  'train'...
    });
trainer.add({
    'type' 'Silence' ...
    'name' 'silence' ...
    'bottom' 'label' ...
    });


trainer.flowOrder = {'train'};
trainer.repeat     = 100;
trainer.savePath   = fullfile('data','exp');
trainer.gpu        = 1;

trainOp.iter        = 600;  
trainOp.numToSave   = 600*conf.save;  
trainOp.displayIter = 20;
trainOp.lrSteps     = 600;
trainOp.lrGamma     = 0.95;
trainOp.lrPolicy    = @lrPolicy;

trainLayers = trainer.getLayerIDs('data_train', 'RandomDistortion', 'local1', 'lr1', ...
                                  'local2', 'lr2', 'local3', 'lr3', 'local4', 'lr4', ...
                                  'local5', 'reshape', 'localTransform', 'crop', 'loss', 'silence');

trainer.addFlow('train', trainOp, trainLayers);

trainer.run();

end

function res = lrPolicy(globalIterNum, currentPhaseTotalIter, lr, gamma, power, steps)
     %res = lr*(gamma^floor((currentPhaseTotalIter-1)/steps));
     res = lr;
end

function out = biInit(d,~)
    [x,y] = meshgrid(1:32,1:32);
    out = single(reshape(cat(3,x,y),d));
    out = ((out-1)./(32-1)-0.5)*2.0;
end