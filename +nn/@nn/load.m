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
load(dest, 'network');
%merge
if ~onlyWeights && exist('network', 'var')
    obj.data.load(network.data);
    network.data = [];
    obj.randomSeed = network.randomSeed;
    obj.layerNames = network.layerNames;
    obj.layerNamesInd = network.layerNamesInd;

    for i=1:numel(network.layers)
        try
            tmpHandle = str2func(['nn.layers.', network.layers{i}.origParams.type]);
            tmpObj = tmpHandle();
        catch
            tmpHandle = str2func(network.layers{i}.origParams.type);
            tmpObj = tmpHandle();
        end
        origDisableConnectData = tmpObj.disableConnectData;
        tmpObj.disableConnectData = true;
        tmpObj.net = obj;
        tmpObj.load(network.layers{i});
        tmpObj.disableConnectData = origDisableConnectData;
        if obj.gpu
            tmpObj.moveTo('GPU');
        end
        obj.layers{i} = tmpObj;
    end
    clearvars network;
else
    preservedName = network.data.names(network.data.preserve);
    for ff=preservedName
        f = ff{1};
        newid = obj.data.namesInd.(f);
        oldid = network.data.namesInd.(f);
        obj.data.val{newid} = network.data.val{oldid};
        obj.data.momentum{newid} = network.data.momentum{oldid};
    end
    clearvars network;
end
if obj.gpu
    obj.data.moveTo('GPU');
else
    obj.data.moveTo('CPU');
end

fprintf('done.\n');

end