function dispFlow(obj, layers, varargin)
%DISPFLOW show flow
% varargin{1} = logical value. default = true = show flow(also do setup), false = not show, do setup only
    if ~isempty(varargin)
        assert(numel(varargin)==1);
        v = varargin{1};
    else
        v = true;
    end

    dataSizes = cell(size(obj.data.val));
    for i = layers
        % -------
        if v
            for b=1:numel(obj.layers{i}.bottom)
                tt = dataSizes{obj.layers{i}.bottom(b)};
                fprintf('Layer(''%s'') <- %s [%d, %d, %d, %d]\n', obj.layers{i}.name, obj.data.names{obj.layers{i}.bottom(b)}, tt(1),tt(2),tt(3),tt(4));
            end
        end
        % -------
        if ~obj.layers{i}.didSetup
            tmpSizes = obj.layers{i}.setup(dataSizes(obj.layers{i}.bottom));
        else
            tmpSizes = obj.layers{i}.outputSizes(dataSizes(obj.layers{i}.bottom));
        end
        dataSizes(obj.layers{i}.top) = tmpSizes;
        
        % -------
        if v
            for t=1:numel(obj.layers{i}.top)
                tt = dataSizes{obj.layers{i}.top(t)};
                fprintf('Layer(''%s'') -> %s [%d, %d, %d, %d]\n', obj.layers{i}.name, obj.data.names{obj.layers{i}.top(t)}, tt(1),tt(2),tt(3),tt(4));
            end
        end
        % -------
    end
end