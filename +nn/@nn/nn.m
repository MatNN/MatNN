classdef nn < handle
    properties
        data;
        pha_opt;

        gpus = [];
        phaseOrder = {'default'};
        repeatTimes = [];
        gpuMode = false;
        globalIter = [];
        expDir = fullfile('data','exp');
    end
    properties(SetAccess = protected)
        gUpdateFunc;
        needReBuild = false;
    end

    properties(SetAccess = {?nn.nn, ?nn.layers.template.BaseLayer}, GetAccess = public)
        net;
        solverGPUFun;
        MaxThreadsPerBlock = 1024;
        subPhaseName = '__sub__';
    end


    methods
        function obj = nn(netName)
            n                   = {};
            n.name              = netName;
            n.layers            = {}; %layer
            n.weights           = {}; %weights stores here
            n.weightsDiff       = {};
            n.weightsDiffCount  = [];
            n.phase             = {}; %A structure, each field is a phase name, eg. net.phase.train. And each field contains the IDs of layers.
            n.noSubPhase        = {};
            
            n.momentum          = {}; % number and size exactly the same as weights
            n.learningRate      = []; % learningRate of each weight
            n.weightDecay       = []; % weight Decay of each weight

            n.weightsNames      = {}; %weights names here                                           eg. {'conv1_w1', 'relu1', ...}
            n.weightsNamesIndex = {}; %Inverted index from weight name to weight index. A struct    eg. net.weightsNamesIndex.conv1_w1 = 1, ...
            n.weightsIsMisc     = []; %Stores res.(***), ***= field name. because there are layers use .weights to store miscs, not weights.

            n.layerNames        = {}; %Each layer's name
            n.layerNamesIndex   = {}; %Inverted index from layer name to layer index. A struct

            obj.net = n;

            % ======================

            obj.data.val   = {};
            obj.data.diff  = {};
            obj.data.diffCount = [];
            obj.data.names = {};
            obj.data.namesInd  = {};
            obj.data.connectId = {};
            obj.data.replaceId = {};
            obj.data.outId = {};
            obj.data.srcId = {};
        end


        build(obj, varargin)
        [dataSizes, otherPhaseDataSizes] = buildPhase(obj, face, varargin)
        [data, net] = forward(obj, data, net, face, opts)
        [data, net] = backward(obj, data, net, face, opts, dzdy)
        [data, net] = fb(obj, data, net, face, layerIDs, opts, dzdy)
        [data, net] = f(obj, data, net, face, opts, inVals)
        [data, net] = b(obj, data, net, face, opts, outDiffs)
        run(obj)
        runPhase(obj, currentFace, currentRepeatTimes, globalIterNum, currentIter)

        
        function setPhaseOrder(obj, varargin)
            assert(numel(varargin)>0);
            obj.phaseOrder = varargin;
        end
        function setRepeat(obj, t)
            assert(t>0);
            obj.repeatTimes = t;
        end
        function setSavePath(obj, ff)
            assert(~isempty(ff));
            obj.expDir = ff;
        end
        function setGpu(obj,varargin)
            val = [varargin{:}];
            if numel(val) == 1
                obj.gpus = val;
                disp(gpuDevice(val));
                obj.moveTo('gpu');
                obj.setupSolver();
            elseif numel(val) > 1
                error('Please use spmd to assign each lab a different gpu id.');
            elseif isempty(val)
                obj.gpus = [];
                obj.moveTo('cpu');
            else
                error('Unknown parameter of setGpu().');
            end
            %reset data
            obj.gpuMode = numel(obj.gpus) > 0;
        end
        function setPhasePara(obj, face, opt_user)
            opt.numToNext          = 100;   % Runs how many iterations to next phase
            opt.numToSave          = 50;    % Runs how many iterations to next save intermediate model
            opt.displayIter        = 10;    % Show info every opt.displayIter iterations
            opt.showFirstIter      = true;  % show first iteration info
            opt.iter_size          = 1;     % number of iterations to accumulate gradients and update weights.
                                            % useful for divide a batch into multiple subbatches (to fit limited memory capacity)

            opt.learningRate       = 0.001; % 0 = no backpropagation
            opt.learningRateGamma  = 0.1;
            opt.learningRatePower  = 0.75;
            opt.learningRateSteps  = 1000;
            opt.learningRatePolicy = @(currentTotalIterNumber, currentPhaseTotalIter, lr, gamma, power, steps) lr*(gamma^floor((currentPhaseTotalIter-1)/steps));
            
            opt.weightDecay        = 0.0005;
            opt.momentum           = 0.9;

            opt.conserveMemory     = false; % true: Delete forward results at each iteration, but runs slightly slower
            opt.backpropToLayer    = []; % set this to a layer name, if you want to back propagate up to the specified layer.


            if isfield(opt_user, 'layerSettings')
                opt.layerSettings = opt_user.layerSettings; % you can design your own options for custom layer
            end

            opt.plotDiagnostics = false;
            opt = vl_argparse(opt, opt_user);

            if isempty(opt.numToNext)
                error('You must set opt.numToNext.');
            end

            % set opts
            obj.pha_opt.(face) = opt;
        end


        function net = updateWeightGPU(obj, net, lr, weightDecay, momentum, iter_size, updateWeightsInd)
            gf = obj.solverGPUFun;
            %mb = obj.MaxThreadsPerBlock;
            for w = updateWeightsInd
                len = numel(net.weights{w});
                %gf.GridSize = ceil( len/mb );
                [net.momentum{w}, net.weights{w}] = feval(gf, momentum, net.momentum{w}, lr, net.learningRate(w), weightDecay, net.weightDecay(w), net.weights{w}, net.weightsDiff{w}, iter_size, len);
            end
        end
        function net = updateWeightCPU(obj, net, lr, weightDecay, momentum, iter_size, updateWeightsInd)
            for w = updateWeightsInd
                thisDecay = weightDecay * net.weightDecay(w);
                thisLR = lr * net.learningRate(w);
                net.momentum{w} = momentum * net.momentum{w} - thisLR * (thisDecay*net.weights{w} + net.weightsDiff{w}/iter_size);
                net.weights{w}  = net.weights{w} + net.momentum{w};
            end
        end
        function setupSolver(obj)
            ptxp = [];
            cup = [];
            if isempty(ptxp)
                fileP = fileparts(mfilename('fullpath'));
                ptxp = fullfile(fileP, 'private', 'SGD.ptx');
                cup = fullfile(fileP, 'private', 'SGD.cu');
            end
            obj.solverGPUFun = nn.utils.gpu.createHandle(1, ptxp, cup, 'SGD');
            d = gpuDevice();
            obj.MaxThreadsPerBlock = d.MaxThreadsPerBlock;
            obj.solverGPUFun.ThreadBlockSize = obj.MaxThreadsPerBlock;
        end
        function add(obj, varargin)
            % Accpet two kinds of input
            % 1.
            % add('type','conv','name','conv1',...)
            % 2.
            % add({'type' 'conv' 'name' 'conv1' ...})
            tmpLayer = obj.cellLayer2StructLayer(varargin{:});
            tmpLayer.subPhase = [];
            obj.addLayer(tmpLayer);
        end
        function attach(obj, toLayerNames, varargin)
            % Accpet two kinds of input
            % 1.
            % attach({ParentLayerNames}, 'type','conv','name','conv1',...)
            % 2.
            % attach({ParentLayerNames}, {'type' 'conv' 'name' 'conv1' ...})
            %
            % toLayerNames can be a string or a cell array of strings
            layerNames = {};
            for l=1:numel(obj.net.layers)
                layerNames = [layerNames, obj.net.layers{l}.name];
            end
            tmpLayer = obj.cellLayer2StructLayer(varargin{:});
            tmpLayer.subPhase = [];
            if ~isfield(tmpLayer, 'phase')
                tmpLayer.phase = {};
            end
            if iscell(toLayerNames)
                for i=1:numel(toLayerNames)
                    if all(cellfun('isempty',strfind(layerNames,toLayerNames{i}))==false)
                        error(['Attach to non-existed layer:', toLayerNames{i}]);
                    end
                    %depthNo = getAttachNo(toLayerNames{i})+1;
                    %subName = [toLayerNames{i}, obj.subPhaseName, num2str(depthNo)];
                    subName = [toLayerNames{i}, obj.subPhaseName];
                    tmpLayer.phase = [tmpLayer.phase, subName];
                    setSubPhaseToLayer(toLayerNames{i});
                end
            else
                if all(cellfun('isempty',strfind(layerNames,toLayerNames))==false)
                    error(['Attach to non-existed layer:', toLayerNames]);
                end
                %depthNo = getAttachNo(toLayerNames)+1;
                %subName = [toLayerNames, obj.subPhaseName, num2str(depthNo)];
                subName = [toLayerNames, obj.subPhaseName];
                tmpLayer.phase = [tmpLayer.phase, subName];
                setSubPhaseToLayer(toLayerNames);
            end
            obj.addLayer(tmpLayer);

            % function depthNo = getAttachNo(layerName)
            %     layerPhase = {};
            %     for j=1:numel(obj.net.layers)
            %         if strcmp(obj.net.layers{j}.name, layerName)
            %             layerPhase = obj.net.layers{j}.phase;
            %             break;
            %         end
            %     end
            %     parentIsSub = ~cell('isempty', strfind(layerPhase, obj.subPhaseName));
            %     if all(parentIsSub==false)
            %         depthNo = 0;
            %         return;
            %     else
            %         if sum(parentIsSub) > 1
            %             error('Something wrong, there should be only 1 sub phase per layer.');
            %         else
            %             thePhase = layerPhase(parentIsSub);
            %             thePhase = thePhase{1};
            %             depthNo = str2double(  thePhase(strfind(thePhase, obj.subPhaseName)+numel(obj.subPhaseName):end)  );
            %         end
            %     end
            % end
            function setSubPhaseToLayer(layerName)
                for j=1:numel(obj.net.layers)
                    if strcmp(obj.net.layers{j}.name, layerName)
                        obj.net.layers{j}.subPhase = [layerName, obj.subPhaseName];
                        fprintf('Set Layer: %s.phase = %s', layerName, obj.net.layers{j}.subPhase);
                        break;
                    end
                end
            end

        end
        function removeLayer(obj, name)
            obj.net.layers{obj.net.layerNamesIndex.(name)} = {};
            obj.needReBuild = true;
        end
        % function setLayer(obj, name, BaseLayerObj)
        %     if isa(BaseLayerObj, 'nn.layers.template.BaseLayer')
        %         obj.net.layers{obj.net.layerNamesIndex.(name)}.obj = BaseLayerObj;
        %     else
        %         error('Input must be an nn.layers.template.BaseLayer object.');
        %     end
        %     obj.needReBuild = true;
        % end
        function layerObj = getLayer(obj, name)
            layerObj = obj.net.layers{obj.net.layerNamesIndex.(name)}.obj;
        end

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
            else
                obj.build();
                if exist('data', 'var')
                    clearvars data;
                end
                fprintf('===================================\n');
                for i=1:numel(network.weights)
                    name = network.weightsNames{i};
                    if isequal(size(obj.net.weights{obj.net.weightsNamesIndex.(name)}), network.weights{i})
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
        function save(obj, varargin)
            if numel(varargin)==0
                dest = fullfile(obj.expDir, [obj.net.name, '.mat']);
            else
                dest = varargin{1};
            end
            fprintf('Saving network to %s....', dest);
            [a,~] = fileparts(dest);
            if ~exist(a, 'dir')
                mkdir(a);
            end
            backupnet = obj.net;
            backupdata = obj.data;
            obj.moveTo('CPU');
            network = obj.net;
            data = obj.data;
            for i=1:numel(obj.net.layers)
                network.layers{i}.obj = network.layers{i}.obj.save();
            end
            data.val = cell(size(data.val));
            data.diff = cell(size(data.diff));
            save(dest, 'network', 'data');

            clearvars network data;
            obj.net = backupnet;
            obj.data = backupdata;
            clearvars backupnet backupdata;
            fprintf('done.\n');
        end
        function p = saveFilePath(obj, iter)
            p = fullfile(obj.expDir, sprintf('%s-Iter%d.mat', obj.net.name, iter));
        end
        function moveTo(obj, varargin)
            if numel(varargin)==0
                if numel(obj.gpus)>0
                    dest = 'gpu';
                else
                    dest = 'cpu';
                end
            else
                dest = varargin{1};
            end
            for i=1:numel(obj.data.val)
                obj.data.val{i} = obj.moveTo_private(dest,obj.data.val{i});
                obj.data.diff{i} = obj.moveTo_private(dest,obj.data.diff{i});
            end
            for i=1:numel(obj.net.weights)
                obj.net.weights{i} = obj.moveTo_private(dest,obj.net.weights{i});
            end
            for i=1:numel(obj.net.momentum)
                obj.net.momentum{i} = obj.moveTo_private(dest,obj.net.momentum{i});
            end
            obj.net.learningRate = obj.moveTo_private(dest,obj.net.learningRate);
            obj.net.weightDecay = obj.moveTo_private(dest,obj.net.weightDecay);
        end
        
    end

    methods (Access=protected)
        [connectId, replaceId, outId, srcId] = setConnectAndReplaceData(obj, phase, blobSizes)

        function v = invertIndex(~, fields)
            v = struct();
            for i=1:numel(fields)
                v.(fields{i}) = i;
            end
        end
        function tmpLayer = cellLayer2StructLayer(obj, varargin)
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
                obj.addLayer(tmpLayer);
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
        end
        function addLayer(obj, tmpLayer)
            if any(strcmp(tmpLayer.name,obj.net.layerNames))
                error('Layers must have different names.');
            end
            try
                tmpHandle = str2func(['nn.layers.', tmpLayer.type]);
                tmpLayer.obj = tmpHandle();
            catch
                tmpHandle = str2func(tmpLayer.type);
                tmpLayer.obj = tmpHandle();
            end
            obj.net.layers{end+1} = tmpLayer;
            if ~isfield(tmpLayer, 'phase')
                tmpLayer.phase = {};
            end
            obj.needReBuild = true;
        end
        function va = moveTo_private(obj, dest, va)
            if strcmpi(dest, 'gpu')
                if ~isa(va, 'gpuArray')
                    va = gpuArray(single(va));
                end
            elseif strcmpi(dest, 'cpu')
                if isa(va, 'gpuArray')
                    va = gather(va);
                end
            else
                error('Unknown destination.');
            end
        end
        function clearData(obj)
            s = size(obj.data.val);
            obj.data.val   = {};
            obj.data.diff  = {};
            obj.data.diffCount = [];
            obj.data.val   = cell(s);
            obj.data.diff  = cell(s);
            obj.data.diffCount = zeros(s, 'int32');

            s = size(obj.net.weightsDiff);
            obj.net.weightsDiff       = {};
            obj.net.weightsDiffCount  = [];

            obj.net.weightsDiff       = cell(s);
            obj.net.weightsDiffCount  = zeros(s, 'int32');
        end
    end
end