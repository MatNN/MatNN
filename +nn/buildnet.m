function netObj = buildnet(netName, baseNet)
%BUILDNET This function rerank networks, weight names, top names
%
%  USAGE
%  NETOBJ = NN.BUILDNET(NETNAME) creates a netobj with a name
%  NETOBJ = NN.BUILDNET(NETNAME, BASENET) creates a netobj with 
%  a name and a net to be merged. If your layers names are the
%  same as some baseNet.layers{}.name, then those layers will 
%  get the weights which corresponds to the same layer of 
%  baseNet, this is useful for finetuning.
%
%  METHODS
%  NETOBJ.newLayer('key', val, 'key2', val2, ...)
%  NETOBJ.newLayer({'key' val  'key2'  val2 ...})
%  the second form is the same as first form, just for quick typing
%  (no need to put ',' to separate keys).
%  
%  NETOBJ.getNet() process and return your network for nn.train use.
%  NETOBJ.getNet(architecture) preprocess your network with sepcified
%  architecture name, which can be 'default', 'cuda kernel', 
%  'cuda cublas', 'cuda cudnn'. if you not specify an argument, will 
%  use 'default', which is a hybrid code of ordinary array + gpuArray
%  ; 'cuda kernel' means most of the computation will happened on 
%  precompiled CUDA kernels, slightly faster then gpuArray; 
%  'cuda cublas' uses mex files compiled with CUDA kernels and uses 
%  cuBlas to accelerate computation; 'cuda cudnn' uses cuda cublas + 
%  nVIDIA's cuDNN library to accelerate some kind of CNN computation,
%  this is the fastest option.
%
%  NETOBJ.setDataBlobSize(blobName, blobsize)
%  Because we don't have data layers, so you need to specify it here,
%  also works for mutiple data inputs (with different blobName)
%  blobName is equal to the first layer's bottom name
%
%  NETOBJ.setLayerRootFolder(folder)
%  layer's type set to 'nn.layers.???' will have a prefix comes from
%  this. By default, layer folder root will be 'nn.', so all of your
%  layer type will be cap with this value, 
%  eg 'layers.relu' to 'nn.layers.relu'
%
%
%  NOTE:
%  1. you MUST place your layer in processing order.
%     Currently no ordering mechanism provided.
%  2. you SHOULD give all of your tops different names.
%     No checking routine currently.
%  3. 
% 
%  Common layer phrases:
%  'type'
%  'name'
%  'bottom'
%  'top'
%  '***_param'
%  'phase'
%  'rememberOutput'
% 
% 
%  NET Components:
%  net.name
%  net.weights
%  net.weightsNames
%  net.weightsNamesIndex
%  net.weightsShareId
%  net.momentum
%  net.dataLayer
%  net.layers
%  net.layerobjs
%  net.layerNames
%  net.layerNamesIndex
%  net.blobNames
%  net.blobNamesIndex
%  net.blobConnectId
%  

% --------------------------------------------------------------
%                                                     Initialize
% --------------------------------------------------------------

LayerRootFolder = 'nn'; %relative to your path


netObj.newLayer = @newLayer;
netObj.getNet = @getNetwork;
netObj.setDataBlobSize = @setDataBlobSize;
netObj.setLayerRootFolder = @setLayerRootFolder;
netObj.computeMode = @computeMode;


net = {};
net.name         = netName;
net.dataLayer    = {}; %Each field is the data top's size (1x4)
net.layers       = {}; %layer
net.layerobjs    = {}; %layer handles created object
net.weights      = {}; %weights stores here

net.momentum     = {}; % number and size exactly the same as weights
net.learningRate = []; % learningRate of each weight
net.weightDecay  = []; % weight Decay of each weight

net.weightsShareId    = {}; %Indicates which layer ids is share the same weight,          eg. {[1],[2],[3 4], ...}
net.weightsNames      = {}; %weights names here                                           eg. {'conv1_w1', 'relu1', ...}
net.weightsNamesIndex = {}; %Inverted index from weight name to weight index. A struct    eg. net.weightsNamesIndex.conv1_w1 = 1, ...

net.layerNames        = {}; %Each layer's name
net.layerNamesIndex   = {}; %Inverted index from layer name to layer index. A struct

net.blobNames         = {}; %Each top/bottom's name
net.blobNamesIndex    = {}; %Inverted index from top/bottom name to index. A struct
net.blobConnectId     = {}; %Specify a blob(assume it's a top blob) be used in which layers as bottom (so no self layer)

%net.displayLossBlobId = [];

tmp.blobSizes         = {}; % each cell is the blob size
architecture          = 'default';


%NOTICE: variable 'baseNet' have all the element of variable 'net'



% initialize
if nargin == 2
    if ~isempty(baseNet)
        % remove momentum
        net.momentum(:) = {[]};
        %baseNet = [];
    end
else
    baseNet = {};
end


% --------------------------------------------------------------
%                             Set root folder of layer functions
% --------------------------------------------------------------
    function setLayerRootFolder(fo)
        LayerRootFolder = fo;
    end


% --------------------------------------------------------------
%                                                   Remove Phase
% --------------------------------------------------------------
    % 把有設定phase的去掉
    function computeMode(modeName)
        architecture = modeName;
    end

% --------------------------------------------------------------
%                                                 Setup momentum
% --------------------------------------------------------------
    function setupMomentum()
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

        %do blobProcess
        blobProcess();

        %set blobsize cell array
        tmp.blobSizes = cell(size(net.blobNames));

        %set data blob size
        for i=fieldnames(net.dataLayer)'
            if isfield(net.blobNamesIndex, i{1})
                ind = net.blobNamesIndex.(i{1});
                tmp.blobSizes{ind} = net.dataLayer.(i{1});
            else
                warning(['Data layer: ''', i{1}, ''' is not used.']);
            end
        end

        %Replace net.layer.weights to index, points to net.weights
        net.layerobjs = cell(1, numel(net.layers));
        net.weightsNames = {};
        net.weightsShareId = {};
        for i=1:numel(net.layers)
            tmpHandle = str2func([LayerRootFolder,'.', net.layers{i}.type]);
            net.layerobjs{i} = tmpHandle(architecture); % execute layer function!!!
            if ~isempty(net.layers{i}.bottom)
                [res, topSizes, param] = net.layerobjs{i}.setup(net.layers{i}, tmp.blobSizes(net.layers{i}.bottom));
            else
                [res, topSizes, param] = net.layerobjs{i}.setup(net.layers{i}, {});
            end

            % update param for particular layer
            % NOTICE: if you don't want any output other than weghts to be updated, 
            %          make sure param.xxxxx_param.learningRate = 0
            if isstruct(param)
                for fi = fieldnames(param)'
                    net.layers{i}.(fi{1}) = param.(fi{1});
                end
            end

            %set blobsize
            tmp.blobSizes(net.layers{i}.top) = topSizes;

            % if res has multiple fields, means this layer wants to save
            % a lot of things
            resfield = [];
            if isstruct(res)
                resfield = fieldnames(res)';
            else
                % skip current layer, because this layer don't need .weight
                continue;

            end

            % although a layer may have outputs other than .weights
            % we still save them to net.weights, becuase net.weights support
            % sharing and solver update.
            for fi = 1:numel(resfield)
                paName = [resfield{fi}, '_param'];
                
                if numel(res.(resfield{fi})) ~= numel(net.layers{i}.(paName).name)
                    error(['numel(',resfield{fi}, ') must equal to numel(',paName,'.name)']);
                end
                replaceWeights = zeros(1, numel(res.(resfield{fi})));
                
                for w=1:numel(res.(resfield{fi}))
                    if ~isempty(net.layers{i}.(paName).name{w})
                        % concatenate user defined name with '_u' to avoid 
                        % conflict of autogenerated name .
                        newName = sprintf([net.layers{i}.(paName).name{w}, '_', resfield{fi} ,'_u'], w);
                    else
                        newName = sprintf([net.layers{i}.name, '_', resfield{fi}, '_%d_auto'], w);
                    end
                    
                    [ind, rep] = checkWeightsNames(net.layers{i}.name, newName, res.(resfield{fi}){w}, param.(paName).learningRate(w), param.(paName).weightDecay(w));
                    %if rep
                        if numel(net.weightsShareId) < ind
                            net.weightsShareId{ind} = i;
                        else
                            net.weightsShareId{ind} = [net.weightsShareId{ind}, i];
                        end
                        
                    %end
                    replaceWeights(w) = ind;
                end

                net.layers{i}.weights = replaceWeights;
            end

        end
        function [ind, replicated] = checkWeightsNames(layername, wname, theWeight, LR, decay)
            replicated = false;
            if ismember(wname, net.weightsNames)
                disp(['Same weights name detected: ''', wname, '''. Only the first encountered layer will initialize the weight,']);
                disp('Will use the same weight, lr and decay. This layer settings will be ignored.');
                ind = net.weightsNamesIndex.(wname);
                replicated = true;
            else
                net.weightsNames = [net.weightsNames, wname];
                ind = numel(net.weightsNames);
                net.weightsNamesIndex.(wname) = ind;
                net.learningRate(ind) = LR;
                net.weightDecay(ind)  = decay;
                %check if in baseNet
                if isfield(baseNet, 'weightsNames') && ismember(wname, baseNet.weightsNames)
                    existWeights = baseNet.weights{baseNet.weightsNamesIndex.(wname)};
                    if ~isequal(size(existWeights), size(theWeight))
                        fprintf('[<strong>FAILED</strong>]  Use existed weight ''%s'' of Layer ''%s''\n          Create a new weight\n', wname ,layername);
                        net.weights{ind} = theWeight;
                    else
                        fprintf('Use existed weight ''%s'' of Layer ''%s''\n', wname ,layername);
                        net.weights{ind} = existWeights;
                    end
                else
                    net.weights{ind} = theWeight;
                end
            end
        end

        clearvars res;

        %find blobConnectId
        net.blobConnectId = cell(1, numel(net.blobNames));
        for i = 1:numel(net.layers)
            if ~isempty(net.layers{i}.bottom)
                for b = 1:numel(net.layers{i}.bottom)
                    net.blobConnectId{net.layers{i}.bottom(b)} = [net.blobConnectId{net.layers{i}.bottom(b)}, i];
                end
            end
        end

        %set lr/weightdecay/momentum
        setupMomentum();

        %return net
        newNet = net;
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
        if isa(varargin{1}, 'cell') && numel(varargin) == 1
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
%                            top/bottom process and check errors
% --------------------------------------------------------------
    function blobProcess()
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
                if ischar(tt)
                    tt = {tt};
                    net.layers{i}.top = tt;
                end
                for j=1:numel(tt)
                    if isempty(tt{j})
                        error('layer top must not be empty.');
                    end
                    tops = [tops, tt(j)];
                end
            end
        end

        for i = 1:NLayers 
            if isfield(net.layers{i}, 'bottom')
                tt = net.layers{i}.bottom;
                if ischar(tt)
                    tt = {tt};
                    net.layers{i}.bottom = tt;
                end
                for j = 1:numel(tt)
                    if isempty(tt{j})
                        error('layer bottom must not be empty.');
                    end
                    bottoms = [bottoms, tt(j)];
                end
            end
        end

        %give numbering the bottoms and tops
        blobNames = unique([bottoms, tops]);
        blobNamesIndex = [];
        link = [];
        linkCount = [];
        blobNamesCount = zeros(1,numel(blobNames));
        for i=1:numel(blobNames)
            blobNamesIndex.(blobNames{i}) = i; %inverted index
            linkCount.(blobNames{i}) = 0;
            link.(blobNames{i}) = [];
        end

        %check names repeated
        uniqueLayerName = unique(names);
        assert(numel(names) == numel(uniqueLayerName), 'You have layers use the same name.');
        clearvars uniqueLayerName;

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
            else
                net.layers{i}.bottom = [];
            end
            t = [];
            if isfield(net.layers{i}, 'top')
                for j=1:numel(net.layers{i}.top)
                    na = blobNamesIndex.(net.layers{i}.top{j});
                    t = [t, na];
                end
                net.layers{i}.top = t;
            else
                net.layers{i}.top = [];
            end

            if ~isfield(net.layers{i}, 'weights')
                net.layers{i}.weights = [];
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