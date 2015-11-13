function [data, net] = fb(obj, data, net, face, opts, dzdy)
    data.diffCount = data.diffCount.*int32(0);

    % FORWARD
    for i = net.phase.(face)
        l = net.layers{i};

        tmp = net.weightsIsMisc(l.weights);
        weightsInd = l.weights(~tmp);
        miscInd = l.weights(tmp);

        [data.val(l.top), net.weights(weightsInd), net.weights(miscInd)] = ...
            l.obj.forward(opts, data.val(l.top), data.val(l.bottom), net.weights(weightsInd), net.weights(miscInd));
    end

    % BACKWARD
    if opts.learningRate ~= 0
        data.diff(data.outId.(face)) = {dzdy};

        for i = net.phase.(face)(end:-1:1)
            l = net.layers{i};
            
            tmp = net.weightsIsMisc(l.weights);
            weightsInd = l.weights(~tmp);
            miscInd = l.weights(tmp);

            if opts.avgGradient
                tmpCount1 = res.dzdxCount(l.top);
                tmpCount1(tmpCount1==0) = 1;
                tmp_input_dzdx = res.dzdx(l.top);
                for yy = 1:numel(l.top)
                    tmp_input_dzdx{yy} = tmp_input_dzdx{yy}./tmpCount1(yy);
                end
                tmpCount2 = res.dzdwCount(weightsInd);
                tmpCount2(tmpCount2==0) = 1;
                tmp_input_dzdw = res.dzdw(weightsInd);
                for yy = 1:numel(weightsInd)
                    tmp_input_dzdw{yy} = tmp_input_dzdw{yy}./tmpCount2(yy);
                end
                [tmpdzdx, tmpdzdw, net.weights(miscInd)] = l.obj.backward(opts, data.val(l.top), data.val(l.bottom), net.weights(weightsInd), net.weights(miscInd), tmp_input_dzdx,  tmp_input_dzdw);
            else
                [tmpdzdx, tmpdzdw, net.weights(miscInd)] = l.obj.backward(opts, data.val(l.top), data.val(l.bottom), net.weights(weightsInd), net.weights(miscInd), data.diff(l.top), net.weightsDiff(weightsInd));
            end

            
            dzdxEmpty = ~cellfun('isempty', tmpdzdx);

            for b = find(dzdxEmpty)
                if any(data.connectId.(face){l.bottom(b)} == i) && ~any(data.replaceId.(face){l.bottom(b)} == i) && data.diffCount(l.bottom(b))
                    data.diff{l.bottom(b)} = data.diff{l.bottom(b)} + tmpdzdx{b};
                    data.diffCount(l.bottom(b)) = data.diffCount(l.bottom(b))+1;
                else
                    data.diff(l.bottom(b)) = tmpdzdx(b);
                    data.diffCount(l.bottom(b)) = 1;
                end

            end
            
            % be careful of modifying this.
            dzdwEmpty  = ~cellfun('isempty', tmpdzdw); % find d_weights are not empty
            for w = find(dzdwEmpty & net.weightsDiffCount(weightsInd))
                net.weightsDiff{weightsInd(w)} = net.weightsDiff{weightsInd(w)} + tmpdzdw{w};
                net.weightsDiffCount(weightsInd(w)) = net.weightsDiffCount(weightsInd(w))+1;
            end

            dzdwEmpty2 = dzdwEmpty & ~net.weightsDiffCount(weightsInd);
            net.weightsDiff(weightsInd(dzdwEmpty2)) = tmpdzdw(dzdwEmpty2);
            net.weightsDiffCount(weightsInd(dzdwEmpty2)) = 1;

            if strcmp(opts.backpropToLayer, l.name)
                break;
            end
        end
    end
end