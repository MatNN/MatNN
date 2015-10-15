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
%
%  NETOBJ.setLayerRootFolder(folder)
%  layer's type set to 'nn.layers.???' will have a prefix comes 
%  from this. By default, layer folder root will be 'nn.', so 
%  all of your layer type will be cap with this value, 
%  eg 'layers.relu' to 'nn.layers.relu'
%
%
%  NOTE:
%  1. you MUST place your layer in processing order.
%     Currently no ordering mechanism provided.
%  2. you SHOULD give all of your tops different names.
%     No routine checking currently.
%     Unless, you're sure the two layers use the same top name 
%     are direct relative to each other, and is the only parent/child pair
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
%  net.momentum
%  net.layers
%  net.layerobjs
%  net.layerNames
%  net.layerNamesIndex
%  net.blobNames
%  net.blobNamesIndex
%  net.blobConnectId
%  net.replaceId
% 

% --------------------------------------------------------------
%                                                     Initialize
% --------------------------------------------------------------
if ~ischar(netName)
    error('Network name must be char type.');
end

LayerRootFolder = 'nn.layers';


netObj.newLayer = @newLayer;
netObj.setLayers = @setLayers;
netObj.getNet = @getNetwork;
netObj.clear = @clearNet;
netObj.setLayerRootFolder = @setLayerRootFolder;
netObj.setBaseNet = @setBaseNet;
netObj.hash  = @hash; % note, this value will be different if you execute 'getNet()', 'newLayer()', or 'setLayers()'


net                   = {};
net.name              = netName;
net.layers            = {}; %layer
net.layerobjs         = {}; %layer handles created object
net.weights           = {}; %weights stores here
net.phase             = {}; %A structure, each field is a phase name, eg. net.phase.train. And each field contains the IDs of layers.

net.momentum          = {}; % number and size exactly the same as weights
net.learningRate      = []; % learningRate of each weight
net.weightDecay       = []; % weight Decay of each weight

net.weightsNames      = {}; %weights names here                                           eg. {'conv1_w1', 'relu1', ...}
net.weightsNamesIndex = {}; %Inverted index from weight name to weight index. A struct    eg. net.weightsNamesIndex.conv1_w1 = 1, ...
net.weightsIsMisc     = {}; %Stores res.(***), ***= field name. because there are layers use .weights to store miscs, not weights.

net.layerNames        = {}; %Each layer's name
net.layerNamesIndex   = {}; %Inverted index from layer name to layer index. A struct

net.blobNames         = {}; %Each top/bottom's name
net.blobNamesIndex    = {}; %Inverted index from top/bottom name to index. A struct
net.blobConnectId     = {}; %Specify a blob(assume it's a top blob) be used in which layers as bottom (so no self layer)  ***PHASE RELATED***
net.replaceId         = {}; %Specify if any layer uses the same name of bottoms and tops  ***PHASE RELATED***
net.sourceBlobId      = {}; %Specify the data source blob IDs of each phase
net.outputBlobId      = {}; %Specify the output blob IDs of each phase

%net.displayLossBlobId = [];

blobSizes             = {}; % each cell is the blob size      ***PHASE RELATED***, but not saved
backUpNet             = {};


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
%                                                   Set base net
% --------------------------------------------------------------
    function setBaseNet(bn)
        baseNet = bn;
    end

% --------------------------------------------------------------
%                                                 Set all layers
% --------------------------------------------------------------
    function setLayers(layerCells)
        net.layers = layerCells;
    end

% --------------------------------------------------------------
%                             Set root folder of layer functions
% --------------------------------------------------------------
    function setLayerRootFolder(fo)
        LayerRootFolder = fo;
    end


% --------------------------------------------------------------
%             Clear net (must do this before delete this object)
% --------------------------------------------------------------
    function clearNet()
        clearvars net netObj;
    end

% --------------------------------------------------------------
%                                                 Setup momentum
% --------------------------------------------------------------
    function setupMomentum()
        % layers have weights -> must have this three phrases
        % call this function after initialze weights.
        for i = 1:numel(net.weights)
            net.momentum{i} = zeros(size(net.weights{i}), 'single');
        end
    end

% --------------------------------------------------------------
%                              Get hash string of network layers
% --------------------------------------------------------------
    function hashStr = hash()
        [hashStr,~,~]  = nn.utils.hash(net.layers, 'MD5');
    end

% --------------------------------------------------------------
%                 Return network and set internal variable to {}
% --------------------------------------------------------------
    function newNet = getNetwork(opts)
        backUpNet = net;
        hashStr = hash();

        % -----------------------------------------TASK 01 ====>
        % 1. Generate all layers' object
        % 2. Generate phase layer ID sets
        % 3. Put all layer names into "net.layerNames" and setup net.layerNamesIndex
        net.layerobjs       = cell(1, numel(net.layers));
        net.phase           = {};
        net.layerNames      = {}; tmpLayerNames = cell(1, numel(net.layers));
        net.layerNamesIndex = {};

        % 0. get all phase names
        for i=1:numel(net.layers)
            if isfield(net.layers{i}, 'phase')
                if iscell(net.layers{i}.phase)
                    for f = net.layers{i}.phase
                        face = f{1};
                        if ~isfield(net.phase, face)
                            net.phase.(face) = [];
                        end
                    end
                elseif ischar(net.layers{i}.phase)
                    if ~isfield(net.phase, net.layers{i}.phase)
                        net.phase.(net.layers{i}.phase) = [];
                    end
                else
                    error('phase must be a cell of strings or a string.');
                end
            end
        end
        if isempty(net.phase)
            net.phase.train = [];
            net.phase.test  = [];
        end

        opts.hash = hashStr;
        for i=1:numel(net.layers)

            % generate function handle
            if isempty(LayerRootFolder)
                tmpHandle = str2func(net.layers{i}.type);
            else
                tmpHandle = str2func([LayerRootFolder,'.', net.layers{i}.type]);
            end

            % 1. create layer obj
            net.layerobjs{i} = tmpHandle(opts); % execute layer function!!!

            % 2. set layer phase
            if isfield(net.layers{i}, 'phase')
                if iscell(net.layers{i}.phase)
                    for f = net.layers{i}.phase
                        face = f{1};
                        net.phase.(face) = [net.phase.(face), i];
                    end
                elseif ischar(net.layers{i}.phase)
                    net.phase.(net.layers{i}.phase) = [net.phase.(net.layers{i}.phase), i];
                else
                    error('phase must be a cell of strings or a string.');
                end
            else
                for f = fieldnames(net.phase)'
                    face = f{1};
                    net.phase.(face) = [net.phase.(face), i];
                end
            end

            % 3. put layer name
            tmpLayerNames{i} = net.layers{i}.name;
        end

        % 3. unique layer name
        net.layerNames = unique(tmpLayerNames);
        clearvars tmpLayerNames;
        for i=1:numel(net.layerNames)
            net.layerNamesIndex.(net.layerNames{i}) = i;
        end
        % -----------------------------------------TASK 01 ====|


        % -----------------------------------------TASK 02 ====>
        % 1. Setup net.blobNames
        % 2. Setup net.blobNamesIndex
        % 3. Replace net.layers .top/.bottom to id
        % 4. If net.layers{i} has no 'weights' field, create an empty .weights
        blobProcess();

        % 5. Set blob size
        blobSizes = {};
        for f=fieldnames(net.phase)'
            face = f{1};
            blobSizes.(face) = cell(size(net.blobNames));
        end

        % for i=fieldnames(net.dataLayer)'
        %     if isfield(net.blobNamesIndex, i{1})
        %         ind = net.blobNamesIndex.(i{1});
        %         p = net.dataLayer.(i{1}){2};
        %         if isempty(p) % means all phases contain this data blob
        %             for f=fieldnames(net.phase)'
        %                 face = f{1};
        %                 blobSizes.(face){ind} = net.dataLayer.(i{1}){1};
        %             end
        %         else
        %             blobSizes.(p){ind} = net.dataLayer.(i{1}){1};
        %         end
        %     else
        %         warning(['Data layer: ''', i{1}, ''' is not used. Ignore it.']);
        %     end
        % end


        % -----------------------------------------TASK 02 ====|


        % -----------------------------------------TASK 03 ====>
        % 1. Setup net.weightsNames and net.weightsNamesIndex
        % -----------------------------------------TASK 03 ====|

        % -----------------------------------------TASK 04 ====>
        % Replace net.layer.weights to index, points to net.weights
        net.weightsNames = {};
        tmp_weightsFieldNames = {};

        % 1. For each phase
        tmp_isLayerInitialized = false(1, numel(net.layers));
        tmp_layerTopSizeFunc = cell(1, numel(net.layers));
        for f = fieldnames(net.phase)'
            face = f{1};
            fprintf('\nPhase: %s ========================================\n', face);
            tmp_blobSizes = blobSizes.(face);
            for i = net.phase.(face)

                % 2. Check user provided weight
                % 2. Check if required blob is empty(not existed) or not.
                if ~isempty(net.layers{i}.bottom)
                    try
                        tmp_smallSize = tmp_blobSizes(net.layers{i}.bottom);
                    catch err
                        for tmpI=1:numel(net.layers{i}.bottom)
                            if isempty(tmp_blobSizes{tmpI})
                                error(['Layer(''', net.layers{i}.name, ''') requires bottom(''', net.blobNames{tmpI}, '''), but it''s empty.']);
                            end
                        end
                        rethrow(err);
                    end
                    if ~tmp_isLayerInitialized(i)
                        [res, tmp_layerTopSizeFunc{i}, param] = net.layerobjs{i}.setup(net.layers{i}, tmp_smallSize);
                    end

                    if isa(tmp_layerTopSizeFunc{i},'function_handle')
                        topSizes = tmp_layerTopSizeFunc{i}(tmp_smallSize);
                    else
                        topSizes = tmp_layerTopSizeFunc{i};
                    end
                    
                else
                    if ~tmp_isLayerInitialized(i)
                        [res, tmp_layerTopSizeFunc{i}, param] = net.layerobjs{i}.setup(net.layers{i}, {});
                    end

                    if isa(tmp_layerTopSizeFunc{i},'function_handle')
                        topSizes = tmp_layerTopSizeFunc{i}({});
                    else
                        topSizes = tmp_layerTopSizeFunc{i};
                    end
                end


                % 2. Print blob size
                for t = 1:numel(net.layers{i}.bottom)
                    t_ = net.layers{i}.bottom(t);
                    tt = tmp_blobSizes{t_};
                    
                    fprintf('Layer(''%s'') <- %s [%d, %d, %d, %d]\n', net.layers{i}.name, net.blobNames{net.layers{i}.bottom(t)}, tt(1),tt(2),tt(3),tt(4));
                end
                for t = 1:numel(topSizes)
                    tt = topSizes{t};
                    fprintf('Layer(''%s'') -> %s [%d, %d, %d, %d]\n', net.layers{i}.name, net.blobNames{net.layers{i}.top(t)}, tt(1),tt(2),tt(3),tt(4));
                end
                tmp_blobSizes(net.layers{i}.top) = topSizes;


                if ~tmp_isLayerInitialized(i)

                    % 3. If user manually allocates weights,
                    %    user still needs to set weight_param.
                    % 3. If user's weights number == settings weight number,
                    %    then user's weights is valid, save it.
                    if isfield(net.layers{i}, 'weights') && isfield(res, 'weight')
                        if numel(net.layers{i}.weights) == numel(res.weight) || numel(net.layers{i}.weights) == 0
                            for ww=1:numel(net.layers{i}.weights)
                                if isequal(size(net.layers{i}.weights{ww}), size(res.weight{ww}))
                                    res.weight{ww} = net.layers{i}.weights{ww};
                                else
                                    s1 = net.layers{i}.weights{ww};
                                    s2 = res.weight{ww};
                                    s1s = sprintf('[, %d, %d, %d, %d]', size(s1,1), size(s1,2), size(s1,3), size(s1,4));
                                    s2s = sprintf('[, %d, %d, %d, %d]', size(s2,1), size(s2,2), size(s2,3), size(s2,4));
                                    error(['Layer(''', net.layers{i}.name, ''') weights size mismatch, ',s1s,' ~= ',s2s]);
                                end
                            end
                        else
                            error(['Layer(''', net.layers{i}.name, ''') number of weights mismatch, ', num2str(numel(net.layers{i}.weights)), ' ~= ', num2str(numel(res.weight))]);
                        end
                    elseif isfield(net.layers{i}, 'weights') + isfield(res, 'weight') == 1
                        if isfield(net.layers{i}, 'weights') && numel(net.layers{i}.weights) == 0
                            % this means auto added .weights field
                        else
                            error('You provided weights number is not match actual weights number.');
                        end
                    end


                    % 3. Update param for particular layer
                    % NOTICE: if you don't want any output other than weghts to be updated, 
                    %         make sure set param.xxxxx_param.learningRate to 0
                    if isstruct(param)
                        for fi = fieldnames(param)'
                            net.layers{i}.(fi{1}) = param.(fi{1});
                        end
                    end


                    % 5. Check if only .weight and .misc be set.
                    resfield = [];
                    if isstruct(res)
                        resfield = fieldnames(res)';
                        for fi = 1:numel(resfield)
                            if ~strcmp(resfield{fi}, 'weight') && ~strcmp(resfield{fi}, 'misc')
                                error('Resource fieldname must be ''weight'' or ''misc''.');
                            end
                        end
                    else
                        % skip current layer, because this layer don't need .weight
                        continue;
                    end


                    % 5. Check existed weights (base net) by weights name.
                    %    Although a layer may have outputs other than .weights
                    %    we still save them to net.weights, becuase net.weights support
                    %    sharing and solver update.
                    for fi = 1:numel(resfield)
                        paName = [resfield{fi}, '_param'];
                        
                        if numel(res.(resfield{fi})) ~= numel(net.layers{i}.(paName).name)
                            error(['Layer(''', net.layers{i}.name, ''') numel(',resfield{fi}, ') must equal to numel(',paName,'.name)']);
                        end
                        replaceWeights = zeros(1, numel(res.(resfield{fi})));
                        
                        for w=1:numel(res.(resfield{fi}))
                            if ~isempty(net.layers{i}.(paName).name{w})
                                newName = net.layers{i}.(paName).name{w};
                            else
                                newName = sprintf([net.layers{i}.name, '_', resfield{fi}, '_%d'], w);
                            end
                            
                            [ind, ~] = checkWeightsNames(net.layers{i}.name, resfield{fi}, newName, res.(resfield{fi}){w}, param.(paName).learningRate(w), param.(paName).weightDecay(w));
                            replaceWeights(w) = ind;
                            
                        end

                        net.layers{i}.weights = replaceWeights;
                    end
                    clearvars replaceWeights;


                else
                    continue;
                end
                tmp_isLayerInitialized(i) = true;

            end
            blobSizes.(face) = tmp_blobSizes;

        end
        
        
        
        % 5.    Setup net.weight, if a weight exist, check size the same?
        % -----------------------------------------TASK 04 ====|

        
        function [ind, replicated] = checkWeightsNames(layername, wfieldname, wname, theWeight, LR, decay)
            replicated = false;
            if ismember(wname, net.weightsNames)
                existWeights = net.weights{net.weightsNamesIndex.(wname)};
                if ~isequal(size(existWeights), size(theWeight))
                    error(['Same weights name detected: ''', wname, ''', but these weights sizes are not equal.']);
                else
                    disp(['Same weights name detected: ''', wname, '''. Be sure your settings of these weights are the same,']);
                    disp('because this function executes your weight settings does not in order. (Acutally affects by phase)');
                    ind = net.weightsNamesIndex.(wname);
                    replicated = true;
                end
                
            else
                net.weightsNames = [net.weightsNames, wname];
                tmp_weightsFieldNames = [tmp_weightsFieldNames, wfieldname];
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


        % Set helper field:
        %   .weightsIsMisc = indicates wich weight is actually weight, or misc data.
        checkWeight = ismember(tmp_weightsFieldNames, {'weight'});
        checkMisc   = ismember(tmp_weightsFieldNames, {'misc'});
        if sum(checkWeight | checkMisc) ~= numel(tmp_weightsFieldNames)
            error('resoure fieldname must be ''weight'' or ''misc''.');
        end
        net.weightsIsMisc = checkMisc;


        % Set helper field:
        %   .blobConnectId
        net.blobConnectId = {};
        net.replaceId     = {};
        net.outputBlobId  = {};
        net.sourceBlobId  = {};
        for f=fieldnames(net.phase)'
            face = f{1};
            net.blobConnectId.(face) = cell(1, numel(net.blobNames));
            net.replaceId.(face)     = cell(1, numel(net.blobNames)); % if any layer's btm and top use the same name
            net.outputBlobId.(face)  = [];
        end
        for f=fieldnames(net.phase)'
            face = f{1};
            currentLayerID = net.phase.(face);
            allBottoms = [];
            allTops    = [];
            for i = currentLayerID
                if ~isempty(net.layers{i}.bottom)
                    allBottoms = [allBottoms, net.layers{i}.bottom]; %#ok
                    for b = 1:numel(net.layers{i}.bottom)
                        if any(net.layers{i}.top == net.layers{i}.bottom(b))
                            net.replaceId.(face){net.layers{i}.bottom(b)} = [net.replaceId.(face){net.layers{i}.bottom(b)}, i];
                        end
                        net.blobConnectId.(face){net.layers{i}.bottom(b)} = [net.blobConnectId.(face){net.layers{i}.bottom(b)}, i];
                    end
                end
                if ~isempty(net.layers{i}.top)
                    allTops = [allTops, net.layers{i}.top];    %#ok
                end
            end

            totalBlobIDsInCurrentPhase = find(~cellfun('isempty', blobSizes.(face)));
            allBottoms = unique(allBottoms);
            net.outputBlobId.(face) = totalBlobIDsInCurrentPhase( ~ismember(totalBlobIDsInCurrentPhase, allBottoms) );
            net.sourceBlobId.(face) = totalBlobIDsInCurrentPhase( ~ismember(totalBlobIDsInCurrentPhase, allTops) );
        end

        %set lr/weightdecay/momentum
        setupMomentum();

        %return net
        newNet = net;
        net = backUpNet;
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
        if isa(varargin{1}, 'cell') && numel(varargin) == 1
            in = varargin{1};
        elseif isstruct(varargin{1}) && numel(varargin) == 1
            tmpLayer = varargin{1};
            if ~isfield(tmpLayer, 'name')
                error('Layer name not set.');
            end
            if ~isfield(tmpLayer, 'top') && ~isfield(tmpLayer, 'bottom')
                error('No layer top/bottom.');
            end
            net.layers{end+1} = tmpLayer;
            return;
        elseif ischar(varargin{1}) && numel(varargin) > 1 && mod(numel(varargin)) == 0
            in = varargin;
        else
            error('Input must be a struct, a struct definition or a cell.');
        end

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
        net.layers{end+1} = tmpLayer;
    end

% --------------------------------------------------------------
%                            top/bottom process and check errors
% --------------------------------------------------------------
    function blobProcess()
        NLayers = numel(net.layers);

        %Check layer names
        bottoms = {};
        tops = {};
        for i = 1:NLayers
            if isfield(net.layers{i}, 'top')
                tt = net.layers{i}.top;
                if ischar(tt)
                    tt = {tt};
                    net.layers{i}.top = {convertToValidName(net.layers{i}.top)};
                end
                for j=1:numel(tt)
                    if isempty(tt{j})
                        error('layer top must not be empty.');
                    end
                    tc = convertToValidName(tt{j});
                    net.layers{i}.top{j} = tc;
                    tops = [tops, {tc}];
                end
            end
        end

        for i = 1:NLayers 
            if isfield(net.layers{i}, 'bottom')
                tt = net.layers{i}.bottom;
                if ischar(tt)
                    tt = {tt};
                    net.layers{i}.bottom = {convertToValidName(net.layers{i}.bottom)};
                end
                for j = 1:numel(tt)
                    if isempty(tt{j})
                        error('layer bottom must not be empty.');
                    end
                    tc = convertToValidName(tt{j});
                    net.layers{i}.bottom{j} = tc;
                    bottoms = [bottoms, {tc}];
                end
            end
        end

        % Set ordering of bottoms and tops
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

% --------------------- main function end ---------------------
end


% --------------------------------------------------------------
%                                                     Check loop
% --------------------------------------------------------------
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

% --------------------------------------------------------------
%                  Convert Layer name to valid struct field name
% --------------------------------------------------------------
function stringName = convertToValidName(stringName)
    origName = stringName;
    stringName = strtrim(stringName);
    stringName = strrep(stringName, '-', '_');
    if ~isletter(stringName(1))
        stringName = ['prefix_',stringName(2:end)];
    end
    if ~strcmp(origName, stringName)
        fprintf('Convert ''%s'' to ''%s''\n', origName, stringName);
    end
end