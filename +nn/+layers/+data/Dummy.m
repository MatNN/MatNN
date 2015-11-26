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
        function [data, net] = forward(obj, nnObj, l, opts, data, net)
            p = obj.params.dummy;
            if numel(l.bottom)==0
                sizes = p.sizes;
            else
                sizes = cellfun(@nn.utils.size4D, data.val(l.bottom), 'un', false);
            end
            for i=1:numel(sizes)
                if opts.gpuArray
                    if isempty(p.value{i})
                        data.val{l.top(i)} = gpuArray.randn(sizes{i}, 'single');
                    elseif p.value{i}==0
                        data.val{l.top(i)} = gpuArray.zeros(sizes{i}, 'single');
                    elseif p.value{i}==1
                        data.val{l.top(i)} = gpuArray.ones(sizes{i}, 'single');
                    else
                        data.val{l.top(i)} = gpuArray.ones(sizes{i}, 'single').*p.value{i};
                    end
                else
                    if isempty(p.value{i})
                        data.val{l.top(i)} = randn(sizes{i}, 'single');
                    elseif p.value{i}==0
                        data.val{l.top(i)} = zeros(sizes{i}, 'single');
                    elseif p.value{i}==1
                        data.val{l.top(i)} = ones(sizes{i}, 'single');
                    else
                        data.val{l.top(i)} = ones(sizes{i}, 'single').*p.value{i};
                    end
                end
            end
        end
        function [data, net] = backward(obj, nnObj, l, opts, data, net)
            data = nn.utils.accumulateData(opts, data, l);
        end
        function outSizes = outputSizes(obj, opts, l, inSizes, varargin)
            p = obj.params.dummy;
            if numel(l.bottom)==0
                outSizes = p.size;
            else
                outSizes = inSizes;
            end
        end
        function setParams(obj, l)
            obj.setParams@nn.layers.template.BaseLayer(l);
            p = obj.params.dummy;
            if numel(l.bottom)==0
                assert(numel(p.value)>=1);
                assert(numel(p.size) == numel(p.value));
            else
                assert(numel(p.value)>=1);
                assert(numel(p.size)==0);
            end
        end
        function [outSizes, resources] = setup(obj, opts, l, inSizes, varargin)
            [outSizes, resources] = obj.setup@nn.layers.template.BaseLayer(opts, l, inSizes, varargin{:});
            if numel(l.bottom)==0
                assert(numel(l.top)>=1);
            else
                assert(numel(l.top)==numel(l.bottom));
            end
        end
    end

end
