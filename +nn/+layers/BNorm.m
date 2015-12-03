classdef BNorm < nn.layers.template.BaseLayer & nn.layers.template.hasWeight
%BNORM Batch normalization

    methods (Access = protected)
        function modifyDefaultParams(obj)
            obj.default_weight_param = {
                'name' {'', ''} ...
                'generator' {@nn.generator.constant, @nn.generator.constant} ...
                'generator_param' {{'value', 1}, {'value', 0}} ...
                'learningRate' single([1 1]) ...
                'weightDecay' single([0 0])
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
        function forward(obj, nnObj, l, opts, data, net)
            if ~opts.layerSettings.enableBnorm
                data.val{l.top} = data.val{l.bottom};
            else
                data.val{l.top} = obj.f(data.val{l.bottom}, net.weights{l.weights(1)}, net.weights{l.weights(2)});
            end
        end
        function backward(obj, nnObj, l, opts, data, net)
            if ~opts.layerSettings.enableBnorm
                bottom_diff = data.diff{l.top};
            else
                [bottom_diff, weights_diff{1}, weights_diff{2}] = obj.b(data.val{l.bottom}, data.diff{l.top}, net.weights{l.weights(1)}, net.weights{l.weights(2)});
            end
            nn.utils.accumulateData(opts, data, l, bottom_diff);
            nn.utils.accumulateWeight(net, l.weights, weights_diff{:});
        end
        function resources = createResources(obj, opts, l, inSizes, varargin)
            resources.weight = {[],[]};
            resources.weight{1} = obj.params.weight.generator{1}([1, 1, inSizes{1}(3), 1], obj.params.weight.generator_param{1});
            resources.weight{2} = obj.params.weight.generator{2}([1, 1, inSizes{1}(3), 1], obj.params.weight.generator_param{2});
        end
        function setParams(obj, baseProperties)
            obj.setParams@nn.layers.template.BaseLayer(baseProperties);
            assert(all(obj.params.weight.enable_terms), 'All weights must be enabled.');
        end
        function [outSizes, resources] = setup(obj, opts, baseProperties, inSizes, varargin)
            [outSizes, resources] = obj.setup@nn.layers.template.BaseLayer(opts, baseProperties, inSizes, varargin{:});
            assert(numel(baseProperties.bottom)==1);
            assert(numel(baseProperties.top)==1);
        end
    end
    
end