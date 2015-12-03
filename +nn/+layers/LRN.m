classdef LRN < nn.layers.template.BaseLayer

    properties (SetAccess = protected, Transient)
        default_lrn_param = {
            'window_size' 5        ... 
                  'kappa' 1        ...
                  'alpha' 0.0001/5 ...
                   'beta' 0.75
        };
    end

    methods
        function out = f(~, in, window, kappa, alpha, beta)
            out = vl_nnnormalize(in, [window, kappa, alpha, beta]);
        end
        function in_diff = b(~, in, out_diff, window, kappa, alpha, beta)
            in_diff = vl_nnnormalize(in, [window, kappa, alpha, beta], out_diff);
        end
        function forward(obj, nnObj, l, opts, data, net)
            p = obj.params.lrn;
            data.val{l.top} = obj.f(data.val{l.bottom}, p.window_size, p.kappa, p.alpha, p.beta);
        end
        function backward(obj, nnObj, l, opts, data, net)
            p = obj.params.lrn;
            nn.utils.accumulateData(opts, data, l, obj.b(data.val{l.bottom}, data.diff{l.top}, p.window_size, p.kappa, p.alpha, p.beta));
        end
        function [outSizes, resources] = setup(obj, opts, l, inSizes, varargin)
            [outSizes, resources] = obj.setup@nn.layers.template.BaseLayer(opts, l, inSizes, varargin{:});
            assert(numel(l.bottom)==1);
            assert(numel(l.top)==1);
        end
    end

end
