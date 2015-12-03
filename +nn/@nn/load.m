function load(obj, dest, varargin)
    
if isnumeric(dest)
    obj.globalIter = dest;
    dest = obj.saveFilePath(dest);
else
    obj.globalIter = [];
end
if numel(varargin)==1
    if strcmpi(varargin{1}, 'weights')
        onlyWeights = true;
    elseif strcmpi(varargin{1}, 'all')
        onlyWeights = false;
    end
else
    onlyWeights = false;
end
if ~exist(dest, 'file')
    error('Cannot find saved network file.');
end
fprintf('Load network from %s....', dest);
load(dest, 'network', 'data');
%merge
if (isempty(obj.net.weights) && ~onlyWeights) && (exist('network', 'var') && exist('data', 'var'))
    obj.net = network;
    obj.data = data;
    clearvars network data;
    for i=1:numel(obj.net.layers)
        tmpObj = [];
        try
            tmpHandle = str2func(['nn.layers.', obj.net.layers{i}.type]);
            tmpObj = tmpHandle();
        catch
            tmpHandle = str2func(obj.net.layers{i}.type);
            tmpObj = tmpHandle();
        end
        tmpObj.load(obj.net.layers{i}.obj);
        if numel(obj.gpus)>0
            tmpObj.moveTo('GPU');
        end
        obj.net.layers{i}.obj = tmpObj;

    end
    obj.needReBuild = false;
    obj.setRandomSeed();
else
    obj.build();
    if exist('data', 'var')
        clearvars data;
    end
    fprintf('===================================\n');
    for i=1:numel(network.weights)
        name = network.weightsNames{i};
        if isequal(size(obj.net.weights{obj.net.weightsNamesIndex.(name)}), size(network.weights{i}))
            obj.net.weights{obj.net.weightsNamesIndex.(name)} = network.weights{i};
            obj.net.momentum{obj.net.weightsNamesIndex.(name)} = network.momentum{i};
            fprintf('Replace with loaded weight: %s\n', name);
        else
            fprintf('Did ont replace with weight: %s, size mismatch.\n', name);
        end
    end
    clearvars network;
end
obj.moveTo();
fprintf('done.\n');

end