function build(obj, varargin)
    if numel(varargin)==1 && varargin{1} == true
    elseif obj.needReBuild == true
    elseif obj.needReBuild == false
        disp('Did not rebuild. If you need to force rebuild network, use .build(true)');
        return;
    end
    % data process
    bottoms = {};
    tops = {};
    for i=1:numel(obj.net.layers)
        if isfield(obj.net.layers{i}, 'bottom')
            if iscell(obj.net.layers{i}.bottom)
                bottoms = [bottoms, obj.net.layers{i}.bottom{:}];
            else
                bottoms = [bottoms, obj.net.layers{i}.bottom];
            end
        end
        if isfield(obj.net.layers{i}, 'top')
            if iscell(obj.net.layers{i}.top)
                tops = [tops, obj.net.layers{i}.top{:}];
            else
                tops = [tops, obj.net.layers{i}.top];
            end
        end
    end
    tops    = unique(tops, 'stable');
    bottoms = unique(bottoms, 'stable');
    obj.data.names = unique([bottoms, tops], 'stable');
    obj.data.namesInd = obj.invertIndex(obj.data.names);

    phases = {};
    for i=1:numel(obj.net.layers)
        top = [];
        if isfield(obj.net.layers{i}, 'top')
            if iscell(obj.net.layers{i}.top)
                for t=1:numel(obj.net.layers{i}.top)
                    top(t) = obj.data.namesInd.(obj.net.layers{i}.top{t});
                end
            else
                top = obj.data.namesInd.(obj.net.layers{i}.top);
            end
        end
        btm = [];
        if isfield(obj.net.layers{i}, 'bottom')
            if iscell(obj.net.layers{i}.bottom)
                for b=1:numel(obj.net.layers{i}.bottom)
                    btm(b) = obj.data.namesInd.(obj.net.layers{i}.bottom{b});
                end
            else
                btm = obj.data.namesInd.(obj.net.layers{i}.bottom);
            end
        end
        obj.net.layers{i}.top = top;
        obj.net.layers{i}.bottom = btm;


        if isfield(obj.net.layers{i}, 'phase')
            if iscell(obj.net.layers{i}.phase)
                for f = obj.net.layers{i}.phase
                    face = f{1};
                    phases = [phases, face];
                end
            elseif ischar(obj.net.layers{i}.phase)
                phases = [phases, obj.net.layers{i}.phase];
            else
                error('phase must be a cell of strings or a string.');
            end
        end
        

    end
    phases = unique(phases, 'stable');
    if isempty(phases)
        phases = {'default'};
    end
    obj.net.phase = {};
    for ff = phases
        f = ff{1};
        obj.net.phase.(f) = [];
        for i=1:numel(obj.net.layers)
            if isfield(obj.net.layers{i}, 'phase')
                if iscell(obj.net.layers{i}.phase)
                    for c=1:numel(obj.net.layers{i}.phase)
                        if strcmp(f, obj.net.layers{i}.phase{c})
                            obj.net.phase.(f) = [obj.net.phase.(f), i];
                        end
                    end
                    
                elseif ischar(obj.net.layers{i}.phase)
                    if strcmp(f, obj.net.layers{i}.phase)
                        obj.net.phase.(obj.net.layers{i}.phase) = [obj.net.phase.(obj.net.layers{i}.phase), i];
                    end
                else
                    error('phase must be a cell of strings or a string.');
                end
            else
                obj.net.phase.(f) = [obj.net.phase.(f), i];
            end
        end
    end

    obj.net.layerNames      = cellfun(@(x) x.name, obj.net.layers, 'un', false);
    obj.net.layerNamesIndex = obj.invertIndex(obj.net.layerNames);

    % weights/shared weights
    obj.net.weights           = {};
    obj.net.weightsNames      = {};
    obj.net.weightsNamesIndex = {};
    obj.net.momentum          = {};
    obj.net.learningRate      = [];
    obj.net.weightDecay       = [];
    obj.net.weightsIsMisc     = [];
    tmpDataSizes = {};
    for f = fieldnames(obj.net.phase)'
        face = f{1};
        tmpDataSizes.(face) = cell(size(obj.data.names));
        fprintf('\nPhase: %s ========================================\n', face);
        for i = obj.net.phase.(face)
            if ~obj.net.layers{i}.obj.didSetup
                [tmpDataSizes.(face)(obj.net.layers{i}.top), res] = obj.net.layers{i}.obj.setup(obj, obj.net.layers{i}, tmpDataSizes.(face)(obj.net.layers{i}.bottom));
                obj.net.layers{i}.weights = processWeights(obj.net.layers{i}, res);
            else
                tmpDataSizes.(face)(obj.net.layers{i}.top) = obj.net.layers{i}.obj.outputSizes(obj, tmpDataSizes.(face)(obj.net.layers{i}.bottom));
            end
            for b=1:numel(obj.net.layers{i}.bottom)
                tt = tmpDataSizes.(face){obj.net.layers{i}.bottom(b)};
                fprintf('Layer(''%s'') <- %s [%d, %d, %d, %d]\n', obj.net.layers{i}.name, obj.data.names{obj.net.layers{i}.bottom(b)}, tt(1),tt(2),tt(3),tt(4));
            end
            for t=1:numel(obj.net.layers{i}.top)
                tt = tmpDataSizes.(face){obj.net.layers{i}.top(t)};
                fprintf('Layer(''%s'') -> %s [%d, %d, %d, %d]\n', obj.net.layers{i}.name, obj.data.names{obj.net.layers{i}.top(t)}, tt(1),tt(2),tt(3),tt(4));
            end
        end
    end
    obj.net.momentum    = cellfun(@(w) zeros(size(w),'single'), obj.net.weights, 'un', false);
    obj.net.weightsDiff = cellfun(@(w) zeros(size(w),'single'), obj.net.weights, 'un', false);

    [obj.data.connectId, obj.data.replaceId, obj.data.outId, obj.data.srcId] = setConnectAndReplaceData(obj, obj.net.phase, tmpDataSizes);
    obj.data.val   = cell(size(obj.data.names));
    obj.data.diff  = cell(size(obj.data.names));

    obj.data.diffCount = zeros(size(obj.data.diff), 'int32');
    obj.net.weightsDiffCount = zeros(size(obj.net.weightsDiff), 'int32');
    obj.net.weightsIsMisc = logical(obj.net.weightsIsMisc);

    % process weights
    function [wInd] = processWeights(l, rs)
        wInd = [];
        if ~isstruct(rs)
            return;
        end
        fe = fieldnames(rs)';
        for k=fe
            tf = k{1};
            for j=1:numel(l.obj.params.(tf).name)
                if isempty(l.obj.params.(tf).name{j})
                    wname = sprintf([l.name, '_', tf, '_%d'], j);
                else
                    wname = l.obj.params.(tf).name{j};
                end
                wi = find(strcmp(wname, obj.net.weightsNames));
                if ~isempty(wi)
                    if isequal(size(rs.(tf){j}), size(obj.net.(tf){wi}))
                        fprintf('Use same %s: %s\n', tf, wname);
                    else
                        obj.net.weights{wi} = rs.weight{j};
                        fprintf('Replace %s: %s\n', tf, wname);
                        obj.net.learningRate(wi) = l.obj.params.(tf).learningRate(j);
                        obj.net.weightDecay(wi) = l.obj.params.(tf).weightDecay(j);
                    end
                    wInd = [wInd, wi];
                else
                    wi = numel(obj.net.weights)+1;
                    obj.net.weights{wi} = rs.(tf){j};
                    obj.net.weightsNames{wi} = wname;
                    obj.net.weightsNamesIndex.(wname) = wi;
                    obj.net.learningRate(wi) = l.obj.params.(tf).learningRate(j);
                    obj.net.weightDecay(wi) = l.obj.params.(tf).weightDecay(j);
                    if strcmp(tf, 'weight')
                        obj.net.weightsIsMisc(wi) = false;
                    elseif strcmp(tf, 'misc')
                        obj.net.weightsIsMisc(wi) = true;
                    else
                        error('Unknown param: %s\n', tf);
                    end
                    wInd = [wInd, wi];
                end
            end
        end
    end

    obj.moveTo();
    obj.needReBuild = false;
end