classdef LossLayer < nn.layers.template.BaseLayer

    properties (SetAccess = protected, Transient)
        default_loss_param = {
             'labelIndex_start' single(0)    ...
                    'threshold' realmin('single') ...
                   'accumulate' true  ... % report per-batch loss (false) or avg loss (true), this does not affect backpropagation
                  'loss_weight' single(1) ... % a multiplier to the loss
        };
        % 'accumulate' property is NOT available when you call .f() .b() directly.
    end
    

    methods
        % CPU Forward
        function acc = f(obj, in, label)
        end
        % CPU Backward
        function in_diff = b(obj, in, label)
        end

        % Calc Output sizes
        function outSizes = outputSizes(obj, opts, l, inSizes, varargin)
            outSizes = {[1,1,1,1]};
        end

    end
    

end