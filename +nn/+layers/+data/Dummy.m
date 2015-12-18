classdef Dummy < nn.layers.template.BaseLayer
%DUMMY 
% This layer accepts data, and generates data with the same sizes and each value is specified in param.
% If numel(bottom) == 0, then generate data with specifted values and sizes
% value set to empty = randn

    properties (SetAccess = protected, Transient)
        default_dummy_param = {
            'value' {single(0)} ...
            'size' {[]} ...
        };
    end

    methods
        function f(~, varargin)
            error('Not supported.');
        end
        function b(~, varargin)
            error('Not supported.');
        end
        function forward(obj)
            p = obj.params.dummy;
            data = obj.net.data;
            if numel(obj.bottom)==0
                sizes = p.sizes;
            else
                sizes = cellfun(@nn.utils.size4D, data.val(obj.bottom), 'un', false);
            end
            for i=1:numel(sizes)
                if obj.net.opts.gpu
                    if isempty(p.value{i})
                        data.val{obj.top(i)} = gpuArray.randn(sizes{i}, 'single');
                    elseif p.value{i}==0
                        data.val{obj.top(i)} = gpuArray.zeros(sizes{i}, 'single');
                    elseif p.value{i}==1
                        data.val{obj.top(i)} = gpuArray.ones(sizes{i}, 'single');
                    else
                        data.val{obj.top(i)} = gpuArray.ones(sizes{i}, 'single').*p.value{i};
                    end
                else
                    if isempty(p.value{i})
                        data.val{obj.top(i)} = randn(sizes{i}, 'single');
                    elseif p.value{i}==0
                        data.val{obj.top(i)} = zeros(sizes{i}, 'single');
                    elseif p.value{i}==1
                        data.val{obj.top(i)} = ones(sizes{i}, 'single');
                    else
                        data.val{obj.top(i)} = ones(sizes{i}, 'single').*p.value{i};
                    end
                end
            end
            data.forwardCount(obj.bottom, obj.top);
        end
        function backward(obj)
            obj.net.data.backwardCount(obj.bottom,  obj.top);
        end
        function outSizes = outputSizes(obj, inSizes)
            p = obj.params.dummy;
            if numel(obj.bottom)==0
                outSizes = p.size;
            else
                outSizes = inSizes;
            end
        end
        function setParams(obj)
            obj.setParams@nn.layers.template.BaseLayer();
            p = obj.params.dummy;
            if numel(obj.bottom)==0
                assert(numel(p.value)>=1);
                assert(numel(p.size) == numel(p.value));
            else
                assert(numel(p.value)>=1);
                assert(numel(p.size)==0);
            end
        end
        function outSizes = setup(obj, inSizes)
            outSizes = obj.setup@nn.layers.template.BaseLayer(inSizes);
            if numel(obj.bottom)==0
                assert(numel(obj.top)>=1);
            else
                assert(numel(obj.top)==numel(obj.bottom));
            end
        end
    end

end
