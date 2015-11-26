function [data, net] = fb(obj, data, net, face, layerIDs, opts, dzdy)
    data.diffCount = data.diffCount.*int32(0);

    for i = layerIDs
        l = net.layers{i};
        l.no = i;
        [data, net] = l.obj.forward(obj, l, opts, data, net);
    end

    if opts.learningRate ~= 0
        data.diff(data.outId.(face)) = {dzdy};
        for i = layerIDs(end:-1:1)
            l = net.layers{i};
            l.no = i;
            [data, net] = l.obj.backward(obj, l, opts, data, net);
            if strcmp(opts.backpropToLayer, l.name)
                break;
            end
        end
    end
end