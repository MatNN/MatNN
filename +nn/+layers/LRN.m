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
            in_diff = vl_nnnormalize(in, [window, kappa, alpha, beta], out_diff) ;
        end
        function [top, weights, misc] = forward(obj, opts, top, bottom, weights, misc)
            p = obj.params.lrn;
            top{1} = obj.f(bottom{1}, p.window_size, p.kappa, p.alpha, p.beta);
        end
        function [bottom_diff, weights_diff, misc] = backward(obj, opts, top, bottom, weights, misc, top_diff, weights_diff)
            p = obj.params.lrn;
            bottom_diff{1} = obj.b(bottom{1}, top_diff{1}, p.window_size, p.kappa, p.alpha, p.beta);
        end
        function [outSizes, resources] = setup(obj, opts, baseProperties, inSizes)
            [outSizes, resources] = obj.setup@nn.layers.template.BaseLayer(opts, baseProperties, inSizes);
            assert(numel(baseProperties.bottom)==1);
            assert(numel(baseProperties.top)==1);
        end
    end

end
