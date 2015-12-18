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
        function forward(obj)
            p = obj.params.lrn;
            data = obj.net.data;
            data.val{obj.top} = obj.f(data.val{obj.bottom}, p.window_size, p.kappa, p.alpha, p.beta);
            data.forwardCount(obj.bottom, obj.top);
        end
        function backward(obj)
            p = obj.params.lrn;
            data = obj.net.data;
            data.backwardCount(obj.bottom,  obj.top, obj.b(data.val{obj.bottom}, data.diff{obj.top}, p.window_size, p.kappa, p.alpha, p.beta));
        end
        function outSizes = setup(obj, inSizes)
            outSizes = obj.setup@nn.layers.template.BaseLayer(inSizes);
            assert(numel(obj.bottom)==1);
            assert(numel(obj.top)==1);
        end
    end

end
