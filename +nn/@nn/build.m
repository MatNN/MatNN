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
auxs = {};
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
    if isfield(obj.net.layers{i}, 'aux')
        if iscell(obj.net.layers{i}.aux)
            auxs = [auxs, obj.net.layers{i}.aux{:}];
        else
            auxs = [auxs, obj.net.layers{i}.aux];
        end
    end
end
tops    = unique(tops, 'stable');
bottoms = unique(bottoms, 'stable');
auxs    = unique(auxs, 'stable');
obj.data.names = unique([bottoms, tops, auxs], 'stable');
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
    aux = [];
    if isfield(obj.net.layers{i}, 'aux')
        if iscell(obj.net.layers{i}.aux)
            for a=1:numel(obj.net.layers{i}.aux)
                aux(a) = obj.data.namesInd.(obj.net.layers{i}.aux{a});
            end
        else
            aux = obj.data.namesInd.(obj.net.layers{i}.aux);
        end
    end
    obj.net.layers{i}.top = top;
    obj.net.layers{i}.bottom = btm;
    obj.net.layers{i}.aux = aux;

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
rootPhaseInd = cellfun(@(s) isempty(strfind(s, obj.subPhaseName)), phases);
rootPhase = phases(rootPhaseInd);
if isempty(phases)
    phases = {'default'};
    rootPhase = {'default'};
elseif isempty(rootPhase)
    phases = ['default', phases];
    rootPhase = {'default'};
end

obj.net.phase = {};
obj.net.noSubPhase = {};
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
            if isempty(strfind(f, obj.subPhaseName))
                obj.net.phase.(f) = [obj.net.phase.(f), i];
            end
        end
    end
end
obj.net.noSubPhase = obj.net.phase;
tmpAllSubPhaseLayerIDs = [];
for ff = phases
    f = ff{1};
    if ~isempty(strfind(f, obj.subPhaseName))
        tmpAllSubPhaseLayerIDs = [tmpAllSubPhaseLayerIDs, obj.net.phase.(f)]; %#ok
        obj.net.noSubPhase = rmfield(obj.net.noSubPhase, f);
    end
end
for ff = phases
    f = ff{1};
    if isempty(strfind(f, obj.subPhaseName))
        ids = obj.net.noSubPhase.(f);
        obj.net.noSubPhase.(f) = ids(~ismember(ids, tmpAllSubPhaseLayerIDs));
    end
end


obj.net.layerNames      = cellfun(@(x) x.name, obj.net.layers, 'un', false);
obj.net.layerNamesIndex = obj.invertIndex(obj.net.layerNames);

[obj.data.connectId, obj.data.replaceId, obj.data.outId, obj.data.srcId] = setConnectAndReplaceData(obj, obj.net.phase);
obj.data.val   = cell(size(obj.data.names));
obj.data.diff  = cell(size(obj.data.names));
obj.data.diffCount = zeros(size(obj.data.diff), 'int32');

% weights/shared weights
obj.net.weights           = {};
obj.net.weightsNames      = {};
obj.net.weightsNamesIndex = {};
obj.net.momentum          = {};
obj.net.learningRate      = [];
obj.net.weightDecay       = [];
obj.net.weightsIsMisc     = [];
tmpDataSizes = {};
for f = rootPhase
    face = f{1};
    [tmpDataSizes.(face), otherPhaseSizes] = obj.buildPhase(face);
    if ~isempty(otherPhaseSizes)
        for subf = fieldnames(otherPhaseSizes)';
            subface = subf{1};
            tmpDataSizes.(subface) = otherPhaseSizes.(subface);
        end
    end
end

obj.net.momentum    = cellfun(@(w) zeros(size(w),'single'), obj.net.weights, 'un', false);
obj.net.weightsDiff = cellfun(@(w) zeros(size(w),'single'), obj.net.weights, 'un', false);
obj.net.weightsDiffCount = zeros(size(obj.net.weightsDiff), 'int32');
obj.net.weightsIsMisc = logical(obj.net.weightsIsMisc);

obj.moveTo();
obj.needReBuild = false;
obj.setRandomSeed();

end


function [connectId, replaceId, outId, srcId] = setConnectAndReplaceData(obj, phase)

connectId = {};
replaceId     = {};
outId  = {};
srcId  = {};
for f=fieldnames(phase)'
    face = f{1};
    connectId.(face) = cell(1, numel(obj.data.names));
    replaceId.(face)     = cell(1, numel(obj.data.names)); % if any layer's btm and top use the same name
    outId.(face)  = [];
end
for f=fieldnames(phase)'
    face = f{1};
    currentLayerID = phase.(face);
    allBottoms = [];
    allTops    = [];
    srcs = [];
    outs = [];
    for i = currentLayerID
        if ~isempty(obj.net.layers{i}.bottom)
            allBottoms = [allBottoms, obj.net.layers{i}.bottom]; %#ok
            for b = 1:numel(obj.net.layers{i}.bottom)
                btm = obj.net.layers{i}.bottom(b);
                if any(obj.net.layers{i}.top == btm)
                    replaceId.(face){btm} = [replaceId.(face){btm}, i];
                end
                connectId.(face){btm} = [connectId.(face){btm}, i];

                % only add first encountered data
                if all(srcs ~= btm) && all(allTops ~= btm)
                    srcs = [srcs, btm]; %#ok
                end

                %delete used tops
                outs = outs(~ismember(outs, btm));
            end
            
        end
        if ~isempty(obj.net.layers{i}.top)
            allTops = [allTops, obj.net.layers{i}.top]; %#ok
            outs = [outs, obj.net.layers{i}.top]; %#ok
        end
    end

    %totalBlobIDsInCurrentPhase = find(~cellfun('isempty', blobSizes.(face)));
    %allBottoms = unique(allBottoms);
    %outId.(face) = totalBlobIDsInCurrentPhase( ~ismember(totalBlobIDsInCurrentPhase, allBottoms) );
    %srcId.(face) = totalBlobIDsInCurrentPhase( ~ismember(totalBlobIDsInCurrentPhase, allTops) );
    outId.(face) = outs;
    srcId.(face) = srcs;
end

end