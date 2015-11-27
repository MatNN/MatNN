function [data, net] = backward(obj, data, net, face, opts)
    % this method does not reset data operation counts
    %data.diffCount = data.diffCount.*int32(0);

    if opts.learningRate ~= 0
        %data.diff(data.outId.(face)) = outDiffs;
        for i = net.phase.(face)(end:-1:1)
            l = net.layers{i};
            l.no = i;
            [data, net] = l.obj.backward(obj, l, opts, data, net);
            if strcmp(opts.backpropToLayer, l.name)
                break;
            end
        end
    end
    
end