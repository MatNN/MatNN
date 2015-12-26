classdef BNorm < nn.layers.template.WeightLayer
%BNORM Batch normalization

    methods (Access = protected)
        function modifyDefaultParams(obj)
            obj.default_weight_param = {
                'name' {'', '', ''} ...
                'generator' {@nn.generator.constant, @nn.generator.constant, @nn.generator.constant} ...
                'enable_terms' [true, true, true] ... 
                'generator_param' {{'value', 1}, {'value', 0}, {'value', 0}} ...
                'learningRate' single([2 1 0.05]) ...
                'weightDecay' single([0 0 0])
            };
        end
    end
    methods
        function out = f(~, in, w1, w2)
            out = vl_nnbnorm(in, w1, w2);
        end
        function [in_diff, w1_diff, w2_diff] = b(~, in, out_diff, w1, w2)
            [ in_diff, w1_diff, w2_diff ] = vl_nnbnorm(in, w1, w2, out_diff);
        end
        function forward(obj)
            net = obj.net;
            data = net.data;
            if ~net.opts.layerSettings.enableBnorm
                data.val{obj.top} = vl_nnbnorm(data.val{obj.bottom}, data.val{obj.weights(1)}, data.val{obj.weights(2)}, 'moments', data.val{obj.weights(3)});
            else
                data.val{obj.top} = vl_nnbnorm(data.val{obj.bottom}, data.val{obj.weights(1)}, data.val{obj.weights(2)});
            end
            data.forwardCount(obj.bottom, obj.top);
            data.forwardCount(obj.weights, []);
        end
        function backward(obj)
            net = obj.net;
            data = net.data;
            [bottom_diff, weights_diff1, weights_diff2, weights_diff3] = vl_nnbnorm(data.val{obj.bottom}, data.val{obj.weights(1)}, data.val{obj.weights(2)}, data.diff{obj.top});
            
            data.backwardCount(obj.bottom, obj.top, bottom_diff);
            data.backwardCount(obj.weights, [], weights_diff1, weights_diff2, weights_diff3);
        end
        function createResources(obj, inSizes)
            obj.createResources@nn.layers.template.WeightLayer(inSizes, [inSizes{1}(3), 1], [inSizes{1}(3), 1], [inSizes{1}(3), 2]);
            obj.net.data.method(obj.weights(3)) = obj.net.data.methodString.average;
        end
        function setParams(obj)
            obj.setParams@nn.layers.template.BaseLayer();
            assert(all(obj.params.weight.enable_terms), 'All weights must be enabled.');
        end
        function outSizes = setup(obj, inSizes)
            outSizes = obj.setup@nn.layers.template.BaseLayer(inSizes);
            assert(numel(obj.bottom)==1);
            assert(numel(obj.top)==1);
        end
    end
    
end