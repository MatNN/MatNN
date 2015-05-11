function [ o ] = example()
%EXAMPLE Layer


% this example layer random generates data for you

o.name         = 'example';
o.generateLoss = false;
o.setup        = @setup;
o.forward      = @forward;
o.backward     = @backward;

default_example_param = {
                         'blobSize'    [100 100, 3, 128]      ...
                        };
default_display_param = {
                         'show_welcome_msg'    true     ...
                         'show_memoryusage'    true     ...
                        };

    function [resource, topSizes, param] = setup(l, bottomSizes)
        % process your parameters
        if isfield(l, 'exmaple_param')
            wp1 = vllab.utils.vararginHelper(default_example_param, l.exmaple_param);
        else
            wp1 = vllab.utils.vararginHelper(default_example_param, default_example_param);
        end
        if isfield(l, 'display_param')
            wp2 = vllab.utils.vararginHelper(default_display_param, l.display_param);
        else
            wp2 = vllab.utils.vararginHelper(default_display_param, default_display_param);
        end
        
        % set modified parameters
        param.example_param = wp1;
        param.display_param = wp2;
        
        % top number
        topNum = numel(l.top);
        
        % generate topSizes
        topSizes = cell(1, topNum);
        topSizes(:) = {wp.blobSizes};
        
    end

    function [outputBlob, weightUpdate] = forward(opts, l, weights, bottomBlob)
        weightUpdate = {};
        outputBlob = cell(1, numel(l.top));
        for i=1:numel(l.top)
            if opts.gpu
                outputBlob{i} = gpuArray.rand(l.example_param.blobSize, 'single');
            else
                outputBlob{i} = rand(l.example_param.blobSize, 'single');
            end
        end
        
        if l.display_param.show_welcome_msg
            disp('this is example data layer :D');
        end
        if l.display_param.show_memoryusage
            vllab.utils.memstat; % no GPU memory stats T_T
        end
    end
                   
    function [outputdzdx, outputdzdw] = backward(opts, l, weights, bottomBlob, dzdy)
        % data layer does not support backward propagation
        outputdzdx = {};
        outputdzdw = {};
    end

end

