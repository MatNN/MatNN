classdef nn < handle
    properties
        data;

        opts;

        flow = {};

        flowOrder = {};

        gpu = [];
        repeat = [];
        globalIter = [];
        savePath = fullfile('data','exp');
        showDate = true;
        clearDataOnStart = true;
        inParallel = false;
        randomSeed = [];
    end

    properties(SetAccess = {?nn.nn, ?nn.layers.template.BaseLayer}, GetAccess = public)
        name          = [];
        layers        = {};
        layerNames    = {}; %Each layer's name
        layerNamesInd = {}; %Inverted index from layer name to layer index. A struct

        solverGPUFun;
        MaxThreadsPerBlock = 1024;
    end

    methods
        function obj = nn(netName)
            obj.data = nn.nnData();
            obj.name = netName;
        end

        dispFlow(obj, layers, varargin)
        load(obj, dest, varargin)
        [holdIDs, topBottomIDs, outputId, inputId] = findDataID(~, data, layers)
        printBar(alignment, msg, varargin)
        run(obj)
        runFlow(obj, flowOpts, flowLayerIDs, currentRepeatTimes, globalIterNum, currentIter)

        function set.flowOrder(obj, v)
            if ischar(v)
                assert(numel(v)>0);
                v = {v};
            elseif iscell(v)
                assert(all(~cellfun('isempty', v)), 'flow name must not be empty.');
            end
            obj.flowOrder = v;
        end
        function set.repeat(obj, t)
            assert(t>0);
            obj.repeat = t;
        end
        function set.savePath(obj, ff)
            assert(~isempty(ff));
            obj.savePath = ff;
        end
        function set.showDate(obj, v)
            assert(islogical(v));
            obj.showDate = v;
        end
        function set.clearDataOnStart(obj, v)
            assert(islogical(v));
            obj.clearDataOnStart = v;
        end
        function set.inParallel(obj, v)
            assert(islogical(v));
            obj.inParallel = v;
        end
        function set.gpu(obj, val)
            if numel(val) == 1
                obj.gpu = val;
                disp(gpuDevice(val));
                obj.data.moveTo('gpu');
                obj.setupSolver();
            elseif numel(val) > 1
                error('Please use spmd to assign each lab a different gpu id.');
            elseif isempty(val)
                obj.gpu = [];
                obj.data.moveTo('cpu');
            else
                error('Unknown parameter to set gpu.');
            end
            %reset data
            obj.setRandomSeed_private();
        end
        function set.randomSeed(obj, val)
            obj.setRandomSeed_private(val);
            obj.randomSeed = val;
        end
        
        function addFlow(obj, name, opt_user, layerIDs, varargin) % varargin{1} = logical, true= run dispFlow(), false = do not run dispFlow()
            opt.iter               = 100;   % Runs how many iterations
            opt.numToSave          = 50;    % Runs how many iterations to next save intermediate model
            opt.displayIter        = 10;    % Show info every opt.displayIter iterations
            opt.showFirstIter      = true;  % show first iteration info
            opt.iter_size          = 1;     % number of iterations to accumulate gradients and update weights.
                                            % useful for divide a batch into multiple subbatches (to fit limited memory capacity)
            opt.lr       = 0.001; % 0 = no backpropagation
            opt.lrGamma  = 0.1;
            opt.lrPower  = 0.75;
            opt.lrSteps  = 1000;
            opt.lrPolicy = @(currentTotalIterNumber, currentPhaseTotalIter, lr, gamma, power, steps) lr*(gamma^floor((currentPhaseTotalIter-1)/steps));
            
            opt.decay              = 0.0005;
            opt.momentum           = 0.9;

            opt.conserveMemory     = false; % true: Delete forward results at each iteration, but runs slower
            opt = nn.utils.vararginHelper(opt, opt_user);

            if isempty(opt.iter)
                error('You must set opt.iter.');
            end
            if opt.conserveMemory && opt.lr == 0
                warning(['Flow(',  name,'): convserveMemory works on backward mode only.']);
            end
            if isfield(opt_user, 'layerSettings')
                opt.layerSettings = opt_user.layerSettings;
            end

            % set opts
            obj.flow.(name).opts = opt;
            obj.flow.(name).opts.name = name;

            obj.flow.(name).layers = layerIDs;
            obj.printBar('c',' Net( %s ) Architecture: ', name);
            obj.copyOpts();
            if isempty(varargin) || varargin{1}==true
                obj.dispFlow(layerIDs);
            elseif varargin{1} == false
                % do nothing
            else
                error('the last parameter must be logical.');
            end
        end
        function rmFlow(obj, flowName)
            if isfield(obj.flow, flowName)
                obj.flow = rmfield(obj.flow, flowName);
            else
                error('Unknown flow name.');
            end
        end
        
        function updateWeightGPU(obj, data, lr, weightDecay, momentum, iter_size, updateWeightsInd, gf, data_len)
            for w = updateWeightsInd
                if data.method(w) == 0
                    [data.momentum{w}, data.val{w}] = feval(gf, momentum, data.momentum{w}, lr, data.lr(w), weightDecay, data.decay(w), data.val{w}, data.diff{w}, iter_size, data_len(w));
                else
                    lrw = data.lr(w);
                    data.val{w} = (1 - lrw)*data.val{w} + lrw * data.diff{w}/iter_size;
                end
            end
        end
        function updateWeightCPU(obj, data, lr, weightDecay, momentum, iter_size, updateWeightsInd)
            for w = updateWeightsInd
                if data.method(w) == 0
                    thisDecay = weightDecay * data.decay(w);
                    thisLR = lr * data.lr(w);
                    data.momentum{w} = momentum * data.momentum{w} - thisLR * (thisDecay*data.val{w} + data.diff{w}/iter_size);
                    data.val{w}  = data.val{w} + data.momentum{w};
                else
                    lrw = data.lr(w);
                    data.val{w} = (1 - lrw)*data.val{w} + lrw * data.diff{w}/iter_size;
                end
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
            % Accpet 3 kinds of input
            % 1.
            % add('type','conv','name','conv1',...)
            % 2.
            % add({'type' 'conv' 'name' 'conv1' ...})
            % 3.
            % add(BaseLayerObj)
            if isa(varargin{1}, 'nn.layers.template.BaseLayer')
                tmpLayer = varargin{1};
            else
                tmpLayer = obj.cellLayer2StructLayer(varargin{:});
            end
            obj.addLayer(tmpLayer);
        end
        function remove(obj, name)
            id = obj.layerNamesInd.(name);
            obj.layers{id} = {};
            obj.layerNamesInd = rmfield(obj.layerNamesInd, name);
            obj.layerNames{id} = '';
        end

        function retrievedIDs = getLayerIDs(obj, varargin)% varargin{:} = layer names
            [~, retrievedIDs] = ismember(varargin, obj.layerNames);
        end

        function p = saveFilePath(obj, iter)
            p = fullfile(obj.savePath, sprintf('%s-Iter%d.mat', obj.name, iter));
        end

        function save(obj, varargin)
            if numel(varargin)==0
                dest = fullfile(obj.savePath, [obj.name, '.mat']);
            else
                dest = varargin{1};
            end
            fprintf('Saving network to %s....', dest);
            [a,~] = fileparts(dest);
            if ~exist(a, 'dir')
                mkdir(a);
            end

            network = {};
            network.data = obj.data.save();
            network.randomSeed = obj.randomSeed;
            network.layers        = {};
            network.layerNames    = obj.layerNames;
            network.layerNamesInd = obj.layerNamesInd;

            for i=1:numel(obj.layers)
                network.layers{i} = obj.layers{i}.save();
            end
            save(dest, 'network');
            clearvars network;
            fprintf('done.\n');
        end
        
    end

    methods (Access=protected)
        addLayer(obj, l)

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
        function setRandomSeed_private(obj, varargin)% no input = not change,
            seeeeed = now;
            if ~isempty(varargin)
                if ~isempty(varargin{1})
                    seeeeed  = varargin{1};
                else
                    if isempty(obj.randomSeed)
                        return;
                    end
                end
            else
                if ~isempty(obj.randomSeed)
                    seeeeed = obj.randomSeed;
                else
                    return;
                end
            end
            sc = RandStream('CombRecursive','Seed',seeeeed);
            RandStream.setGlobalStream(sc);
            if numel(obj.gpu) > 0
                sg = parallel.gpu.RandStream('CombRecursive','Seed',seeeeed);
                parallel.gpu.RandStream.setGlobalStream(sg);
            end
        end
        function copyOpts(obj)
            obj.opts.gpu              = obj.gpu;
            obj.opts.repeat           = obj.repeat;
            obj.opts.globalIter       = obj.globalIter;
            obj.opts.savePath         = obj.savePath;
            obj.opts.showDate         = obj.showDate;
            obj.opts.clearDataOnStart = obj.clearDataOnStart;
            obj.opts.inParallel       = obj.inParallel;
            obj.opts.randomSeed       = obj.randomSeed;
        end
    end
end