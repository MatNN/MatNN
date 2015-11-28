function [dataSizes, otherPhaseDataSizes] = buildPhase(obj, face, varargin)% varargin{1} = dataSizes, {2} = nested layer depth = 1 or 2 or 3 ...
    if isempty(varargin)
        dataSizes = cell(size(obj.data.names));
    else
        dataSizes = varargin{1};
    end
    
    otherPhaseDataSizes = {};
    if numel(varargin)<2
        fprintf(['\nPhase: %s ', repmat('=', [1, max(0,40-numel(face))]), '\n'], face);
        lap = '';
        N = 0;
    else
        lap = repmat(' ', [1, 2*varargin{2}]);
        N = varargin{2};
    end
    for i = obj.net.phase.(face)
        %check if any layers is a sub-phase layer
        
        if ~isfield(obj.net.layers{i}, 'phase')
            currentLayerPhase = {};
        else
            currentLayerPhase = obj.net.layers{i}.phase;
        end
        if isempty(strfind(face, obj.subPhaseName)) && any(~isempty(strfind(currentLayerPhase, obj.subPhaseName)))
            continue;
        end

        % -------
        for b=1:numel(obj.net.layers{i}.bottom)
            tt = dataSizes{obj.net.layers{i}.bottom(b)};
            fprintf('%sLayer(''%s'') <- %s [%d, %d, %d, %d]\n', lap, obj.net.layers{i}.name, obj.data.names{obj.net.layers{i}.bottom(b)}, tt(1),tt(2),tt(3),tt(4));
        end
        % -------

        if ~obj.net.layers{i}.obj.didSetup
            [tmpDataSizes, res] = obj.net.layers{i}.obj.setup(obj, obj.net.layers{i}, dataSizes(obj.net.layers{i}.bottom), obj, dataSizes, N);
            
            obj.net.layers{i}.weights = processWeights(obj, obj.net.layers{i}, res);
        else
            tmpDataSizes = obj.net.layers{i}.obj.outputSizes(obj, obj.net.layers{i}, dataSizes(obj.net.layers{i}.bottom), obj, dataSizes, N);
        end
        
        if iscell(tmpDataSizes)
            dataSizes(obj.net.layers{i}.top) = tmpDataSizes;
        elseif isstruct(tmpDataSizes)
            dataSizes(obj.net.layers{i}.top) = tmpDataSizes.(face);
            ff = fieldnames(tmpDataSizes)';
            ff = setdiff(ff, face);
            for fe = ff
                f = fe{1};
                otherPhaseDataSizes.(f) = tmpDataSizes.(f);
            end
        else
            error('Wrong type of DataSize.');
        end

        % -------
        for t=1:numel(obj.net.layers{i}.top)
            tt = dataSizes{obj.net.layers{i}.top(t)};
            fprintf('%sLayer(''%s'') -> %s [%d, %d, %d, %d]\n', lap, obj.net.layers{i}.name, obj.data.names{obj.net.layers{i}.top(t)}, tt(1),tt(2),tt(3),tt(4));
        end
        % -------
    end
end

% process weights
function [wInd] = processWeights(obj, l, rs)
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
                if strcmp(tf, 'weight') && isequal(size(rs.(tf){j}), size(obj.net.weights{wi}))
                    fprintf('Use same %s: %s\n', tf, wname);
                elseif strcmp(tf, 'misc') && isequal(size(rs.(tf){j}), size(obj.net.weights{wi}))
                    fprintf('Use same %s: %s\n', tf, wname);
                else
                    obj.net.weights{wi} = rs.weight{j};
                    fprintf('Replace %s: %s\n', tf, wname);
                    obj.net.learningRate(wi) = l.obj.params.(tf).learningRate(j);
                    obj.net.weightDecay(wi) = l.obj.params.(tf).weightDecay(j);
                end
                wInd = [wInd, wi]; %#ok
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
                wInd = [wInd, wi]; %#ok
            end
        end
    end
end