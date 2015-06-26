function netObj = buildnet(netName)
%BUILDNET This function rerank networks, weight names, top names
%
% Common layer phrases:
% 'type'
% 'name'
% 'bottom'
% 'top'
% 'learningRate' *
% 'weightDecay' *
% 'stride' *
% 'pad' *
% 'method' *
% 'pool' *
% 'rate' *
% 'param' *
%
%
% NET Components
% net.name
% net.weights
% net.weightsNames
% net.weightsNamesIndex
% net.weightsShareId
% net.momentum
% net.dataLayer
% net.layers
% net.layerobjs
% net.layerNames
% net.layerNamesIndex
% net.blobNames
% net.blobNamesIndex
% net.blobConnectId
% 
%
% Layers Components
% per layer specific param
% weights *
% weights_name *
% name
% type
% learningRate *
% weightDecay *
%
%

% --------------------------------------------------------------
%                                                     Initialize
% --------------------------------------------------------------

netObj.newLayer = @newLayer;
netObj.getNet = @getNetwork;
netObj.setDataBlobSize = @setDataBlobSize;


net = {};
net.name         = netName;
net.dataLayer    = {}; %Each field is the data top's size (1x4)
net.layers       = {}; %layer
net.layerobjs    = {}; %layer handles created object
net.weights      = {}; %weights stores here
net.momentum     = {}; % number and size exactly the same as weights

net.weightsShareId    = {}; %Indicates which layer ids is share the same weight,          eg. {[1],[2],[3 4], ...}
net.weightsNames      = {}; %weights names here                                           eg. {'conv1_w1', 'relu1', ...}
net.weightsNamesIndex = {}; %Inverted index from weight name to weight index. A struct    eg. net.weightsNamesIndex.conv1_w1 = 1, ...

net.layerNames        = {}; %Each layer's name
net.layerNamesIndex   = {}; %Inverted index from layer name to layer index. A struct

net.blobNames         = {}; %Each top/bottom's name
net.blobNamesIndex    = {}; %Inverted index from top/bottom name to index. A struct
net.blobConnectId     = {}; %Specify a blob(assume it's a top blob) be used in which layers as bottom (so no self layer)

net.misc              = {};


tmp.blobSizes         = {}; % each cell is the blob size



% --------------------------------------------------------------
%                                                    Combine net
% --------------------------------------------------------------
    % 比較兩個net,如果任兩個layer所有參數都一樣(除了bottom和top的名字可以不一樣)
    % 那就可以把已經有的weight, misc, copy進來
    % 其他都不用管
    function combine()


% --------------------------------------------------------------
%                                                   Remove Phase
% --------------------------------------------------------------
    % 把
    function removePhase(pha)
        for i=1:numel(net.layers)
            if isfield('phase')
                if strcmpi(net.layers{i}.phase, pha)

                end
            end
        end
    end


% --------------------------------------------------------------
%                                                  Recover Phase
% --------------------------------------------------------------



% --------------------------------------------------------------
%                                                 Setup momentum
% --------------------------------------------------------------
    function setupMoemtum()
        % layers have weights -> must have this three phrases
        % call this function after initialze weights.
        for i = 1:numel(net.weights)
            net.momentum{i} = zeros(size(net.weights{i}), 'single') ;
        end
    end



% --------------------------------------------------------------
%                                        Set data top size (4-D)
% --------------------------------------------------------------
    function setDataBlobSize(dataTopName, blobSize)
        net.dataLayer.(dataTopName) = blobSize;
    end


% --------------------------------------------------------------
%                 Return network and set internal variable to {}
% --------------------------------------------------------------
    function newNet = getNetwork()
        %do rerank (use top/bottom reference)
        rerank();

        %set blobsize cell array
        tmp.blobSizes = cell(size(net.blobNames));

        %set data blob size
        for i=fieldnames(net.dataLayer)'
            ind = net.blobNamesIndex.(i{1});
            tmp.blobSizes{ind} = net.dataLayer.(i{1});
        end

        %Replace net.layer.weights to index, points to net.weights
        net.layerobjs = {};
        net.weightsNames = {};
        net.weightsShareId = {};
        net.misc = cell(1,numel(net.layers));
        for i=1:numel(net.layers)
            tmpHandle = str2func(net.layers{i}.type);
            net.layerobjs{i} = tmpHandle.setup(); % execute layer function!!!
            if isfield(net.layers{i}, 'bottom')
                [res, topSizes] = net.layerobjs{i}(net.layers{i}, tmp.blobSizes(net.layers{i}.bottom));
            else
                [res, topSizes] = net.layerobjs{i}(net.layers{i}, {});
            end

            %set blobsize
            tmp.blobSizes(net.layers{i}.top) = topSizes;

            if isfield(res, 'weights')
                if isfield(net.layers{i}.weight_param, 'name')
                    if numel(res.weights) ~= numel(net.layers{i}.weight_param.name)
                        error('numel(weights) must equal to numel(weight_param.name)');
                    end
                    replaceWeights = zeros(1, numel(res.weights));
                    for w=1:numel(res.weights)
                        [ind, rep] = checkWeightsNames(net.layers{i}.weight_param.name{w}, res.weights{w});
                        if rep
                            net.weightsShareId{ind} = [net.weightsShareId{ind}, i];
                        end
                        replaceWeights(w) = ind;
                    end
                    net.layers{i}.weights = replaceWeights;
                else
                    replaceWeights = zeros(1, numel(res.weights));
                    for w=1:numel(res.weights)
                        newName = sprintf([net.layers{i}.name, '_w%d'], w);
                        [ind, rep] = checkWeightsNames(newName, res.weights{w});
                        if rep
                            net.weightsShareId{ind} = [net.weightsShareId{ind}, i];
                        end
                        replaceWeights(w) = ind;
                    end
                    net.layers{i}.weights = replaceWeights;
                end
            end
            if isfield(res, 'misc')
                net.misc{i} = res.misc;
            end
        end
        function [ind, replicated] = checkWeightsNames(wname, theWeight)
            replicated = false;
            if ismember(wname, net.weightsNames)
                disp(['Same weights name detected: ', wname, 'only the first encountered layer will initialize the weight.']);
                ind = net.weightsNamesIndex.(wname);
                replicated = true;
            else
                net.weightsNames = [net.weightsNames, wname];
                ind = numel(net.weightsNames);
                net.weightsNamesIndex.(wname) = ind;
                net.weights{ind} = theWeight;
            end
        end

        %find blobConnectId
        net.blobConnectId = cell(1, numel(net.blobNames));
        for i = 1:numel(net.layers)
            if isfield(net.layers{i}, 'bottom')
                for b = 1:numel(net.layers{i}.bottom)
                    net.blobConnectId{net.layers{i}.bottom(b)} = [net.blobConnectId{net.layers{i}.bottom(b)}, i];
                end
            end
        end

        %set lr/weightdecay/momentum
        setupMoemtum();

        %return net
        newNet = nn.carrier(net);
        net = {};
    end


% --------------------------------------------------------------
%                                         Create a new net layer
% --------------------------------------------------------------
    function newLayer(varargin)
        % Accpet two kinds of input
        % 1.
        % newLayer('type','conv','name','conv1',...)
        % 2.
        % newLayer({'type' 'conv' 'name' 'conv1' ...})
        hasName = false;
        hasInput_or_output = false;
        in = varargin;
        if isa(varargin{1}, cell) && numel(varargin) == 1
            in = varargin{1};
        end
        lastLayerInd = numel(net.layers);
        tmpLayer = {};
        for i=1:2:numel(in)
            if strcmp(in{i}, 'name')
                hasName = true;
            end
            if strcmp(in{i}, 'top') || strcmp(in{i}, 'bottom')
                hasInput_or_output = true;
            end
            tmpLayer.(in{i}) = in{i+1};
        end
        if ~hasName || ~hasInput_or_output
            error('No layer name or no top/bottom.');
        end
        net.layers{lastLayerInd+1} = tmpLayer;
    end


% --------------------------------------------------------------
%                                        Rerank and check errors
% --------------------------------------------------------------
    function rerank()
        NLayers = numel(net.layers);

        %Check layer names
        names = cell(1,NLayers);
        bottoms = {};
        tops = {};
        for i = 1:NLayers
            if isfield(net.layers{i}, 'name')
                names{i} = net.layers{i}.name;
                if isempty(names{i})
                    error('layer name must not be empty.');
                end
            else
                error('Each layer must have a name.');
            end
            if isfield(net.layers{i}, 'top')
                tt = net.layers{i}.top;
                for j=1:numel(tt)
                    if isempty(tt{j})
                        error('layer top must not be empty.');
                    end
                    tops = [tops, tt(j)];
                end
            end
        end

        noFirstLayerBottomsIndex = 1;
        for i = 1:NLayers 
            if isfield(net.layers{i}, 'bottom')
                tt = net.layers{i}.bottom;
                for j = 1:numel(tt)
                    if isempty(tt{j})
                        error('layer bottom must not be empty.');
                    end
                    bottoms = [bottoms, tt(j)];
                end
            end
            if i == 1
                noFirstLayerBottomsIndex = numel(bottoms);
            end
        end

        [~,boo] = ismember(bottoms, tops); %because we don't have data layers, so don't count first layer's bottom
        if any(boo==false)
            warning('bottoms and tops mismatch.');
        end
        utops = unique(tops);
        if numel(utops) ~= numel(tops)
            error('name of top must be unique.');
        end

        clearvars utops boo;


        %give numbering the bottoms and tops
        blobNames = unique([tops, bottoms]);
        blobNamesIndex = [];
        link = [];
        linkCount = [];
        blobNamesCount = zeros(1,numel(blobNames));
        for i=1:numel(blobNames)
            blobNamesIndex.(blobNames{i}) = i; %inverted index
            linkCount.(blobNames{i}) = 0;
            link.(blobNames{i}) = [];
        end

        %reorder net structure by top and bottom
        %  create link
        top2LayerInd = [];
        for i=1:NLayers
            if isfield(net.layers{i}, 'top')
                for j=1:numel(net.layers{i}.top)
                    top2LayerInd.(net.layers{i}.top{j}) = i;
                    if isfield(net.layers{i}, 'bottom')
                        link.(net.layers{i}.top{j}) = net.layers{i}.bottom;
                    end
                end
            end
        end
        %  trace
        for i=1:numel(tops)
            linkCount = traceBottom(tops{i}, tops{i}, link, linkCount);
        end
        % Sort layers (descend), use stable sort
        %{
        totalCounts = zeros(1,NLayers);
        for i = fieldnames(top2LayerInd)'
            totalCounts(top2LayerInd.(i{1})) = totalCounts(top2LayerInd.(i{1}))+linkCount.(i{1});
        end
        [~, ind] = sort(totalCounts, 'descend');
        net.layers = net.layers(ind);
        net.layerNames = names(ind);
        %}
        % No sort, because shared weights must be initialized by the first owner.
        % 'The first owner' means the first encountered layer which have weights name 
        % the same as some other layers.
        % So your layer order is important.
        net.layerNames = names;
        for i=1:numel(net.layerNames)
            net.layerNamesIndex.(net.layerNames{i}) = i;
        end

        %Change bottoms and tops to number
        net.blobNames = blobNames;
        net.blobNamesIndex = blobNamesIndex;
        for i=1:NLayers
            b = [];
            if isfield(net.layers{i}, 'bottom')
                for j=1:numel(net.layers{i}.bottom)
                    na = blobNamesIndex.(net.layers{i}.bottom{j});
                    b = [b, na];
                    blobNamesCount(na) = blobNamesCount(na)+1;
                end
                net.layers{i}.bottom = b;
            end
            t = [];
            if isfield(net.layers{i}, 'top')
                for j=1:numel(net.layers{i}.top)
                    na = blobNamesIndex.(net.layers{i}.top{j});
                    t = [t, na];
                end
                net.layers{i}.top = t;
            end
        end
    end


end

function blobCount = traceBottom(startName, ba, link, blobCount)
    la = link.(ba);
    for i=1:numel(la)
        if ~strcmp(startName, la{i})
            blobCount.(la{i}) = blobCount.(la{i})+1;
            blobCount = traceBottom(startName, la{i}, link, blobCount);
        else
            error('You have a counter references bottom <-> top.');
        end
    end
end