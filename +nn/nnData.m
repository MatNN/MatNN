classdef nnData < nn.BaseObject
    properties
        val;
        diff;
        momentum;
        method = [];
        ref;
        currentCount = {};
        maxCount = {};
        preserve = logical([]);
        lr;
        decay;

        conserveMemory = false;
    end
    properties (SetAccess = {?nn.BaseObject}, GetAccess = public)
        names = {};
        namesInd = {}; % struct, each fieldname = var name, field value = var ind
        allInd = [];
        methodString;
    end

    methods
        function obj = nnData()
            obj.methodString.gradient = 0;
            obj.methodString.average = 1;
        end

        function b = isVar(obj, IDorName)
            b = false;
            if ischar(IDorName)
                if isfield(obj.namesInd, IDorName)
                    b = true;
                end
            elseif isreal(IDorName)
                if IDorName > 0 && IDorName <= numel(obj.names)
                    if ~isempty(obj.names{IDorName})
                        b = true;
                    end
                end
            end
        end

        function name = getNameByID(obj, id)
            name = obj.names{id};
        end

        function id = getIDbyName(obj, name)
            id = obj.namesInd.(name);
        end

        function setValue(obj, name, v)
            obj.val{obj.namesInd.(name)} = v;
        end
        function setPartial(obj, IDorName, varargin)
            assert(mod(numel(varargin),2) == 0, 'setPartial(): Format incorrect.');
            if ischar(IDorName)
                id = obj.namesInd.(IDorName);
            else
                id = IDorName;
            end
            for i=1:2:numel(varargin)
                if isprop(obj, varargin{i})
                    if iscell(obj.(varargin{i}))
                        obj.(varargin{i}){id} = varargin{i+1};
                    else
                        obj.(varargin{i})(id) = varargin{i+1};
                    end
                else
                    error('Property %s is not in nnData.', varargin{i});
                end
            end
        end

% ====== ADD / REMOVE VAR =======
        function id = addVar(obj, name, varargin) % ignore already exists var, varargin{1} = flag, show warning of use exists variables
            if ~ismember(name, obj.names)
                if isempty(obj.allInd)
                    pickInds = 1;
                else
                    pickInds = [find(cellfun('isempty',obj.names)), numel(obj.names)+1];
                end
                obj.names{pickInds(1)} = name;
                obj.allInd = [obj.allInd, pickInds(1)];
                obj.namesInd.(name) = pickInds(1);
                obj.initVar(pickInds(1));
            else
                if ~isempty(varargin) && varargin{1} == true
                    fprintf('Use exist var: %s\n', name);
                end
            end
            id = obj.namesInd.(name);
        end
        function removeVar(obj, IDorName)
            if ischar(IDorName)
                if ismember({IDorName}, obj.names)
                    id = obj.namesInd.(IDorName);
                else
                    error('Variable Name: %s doesn''t exist.\n', IDorName);
                end
            elseif isreal(IDorName)
                if ismember(IDorName, obj.allInd)
                    id = IDorName;
                else
                    error('Variable ID: %s doesn''t exist.\n', IDorName);
                end
            else
                error('Unknown ID/Name to remove: %s\n', IDorName);
            end
            obj.initVar(id);
            obj.namesInd = rmfield(obj.namesInd, IDorName);
            obj.names{id} = [];
            obj.allInd = setdiff(obj.allInd,id,'stable');
        end

% ====== HOLD / RELEASE VAR =======
        function holdVar(obj, IDorName)
            if ischar(IDorName)
                id = obj.namesInd.(IDorName);
                obj.ref(id) = obj.ref(id)+1;
            elseif isreal(IDorName)
                obj.ref(IDorName) = obj.ref(IDorName)+1;
            end
        end
        function releaseVar(obj, IDorName)
            if ischar(IDorName)
                id = obj.namesInd.(IDorName);
            elseif isreal(IDorName)
                id = IDorName;
            end
            if obj.ref(id)==0
                error('Alreay release var: %s', IDorName);
            end
            obj.ref(id) = obj.ref(id)-1;
            if obj.ref(id)==0
                obj.removeVar(id)
            end
        end

% ====== ADD / REMOVE COUNT =======
        function forwardCount(obj, fromIds, toIDs) % ids = row vector
            c = obj.currentCount;
            m = obj.maxCount;
            for i=1:numel(fromIds)
                id = fromIds(i);
                if isempty(c{id})
                    c{id} = 1;
                else
                    c{id}(1) = c{id}(1)+1;
                    %fprintf('Var %s Count added!, its ref = %d, count = %d\n', obj.names{id}, obj.ref(id), c{id});
                end
                m{id} = c{id};
            end
            for i=1:numel(toIDs)
                id = toIDs(i);
                c{id} = [0, c{id}];
                m{id} = c{id};
            end
            obj.currentCount = c;
            obj.maxCount = m;
            % use obj.count(ids) = obj.count(ids)+1 will causes repeat ids only add one times.
        end
        function backwardCount(obj, inIds, outIDs, varargin) % varargin = diffs of inIDs
            c = obj.currentCount;
            m = obj.maxCount;
            cM = obj.conserveMemory;
            if cM, deleteID = []; end
            for i=numel(outIDs):-1:1
                id = outIDs(i);
                if isempty(c{id}) || c{id}(1) == 0
                    if numel(c{id})>1
                        c{id} = c{id}(2:end);
                    else
                        c{id} = [];
                        if cM, deleteID = [deleteID, id]; end
                    end
                    m{id} = c{id};
                end
            end
            obj.maxCount = m;

            for i=numel(inIds):-1:1
                id = inIds(i);
                if c{id}(1) > 0 && m{id}(1) > 1
                    if ~isempty(varargin) && ~isempty(varargin{i})
                        if isempty(obj.diff{id}) || m{id}(1) == c{id}(1)
                            obj.diff{id} = varargin{i};
                        else
                            %fprintf('Var %s ACTUALLY added!, its ref = %d, count = %d\n', obj.names{id}, obj.ref(id), c{id});
                            obj.diff{id} = obj.diff{id}+varargin{i};
                        end
                    end
                    c{id}(1) = c{id}(1)-1;
                elseif c{id} == 1
                    if ~isempty(varargin) && ~isempty(varargin{i})
                        obj.diff{id} = varargin{i};
                    end
                    c{id}(1) = c{id}(1)-1;
                else
                    error('Attempt to save diff to an non-avaliable variable');
                end
            end
            if cM
                deleteID = deleteID(~obj.preserve(deleteID));
                obj.val(deleteID) = {[]};
                obj.diff(deleteID) = {[]};
            end
            obj.currentCount = c;
        end
        function clearCount(obj, ids)
            obj.currentCount(ids) = {[]};
            obj.maxCount(ids) = {[]};
        end

% ====== CLEAR / RESET / INIT =======
        function clear(obj, ids, varargin) 
        % varargin{1} = string, 
        % 'force' = force eliminate data, even marked as preserve
        % 'all'   = set all data to init value;
        % note: names, namesInd, allInd are not affected
            if isempty(varargin)
                force = false;
            elseif strcmpi(varargin{1}, 'force')
                force = true;
            elseif strcmpi(varargin{1}, 'all')
                force = 2;
            end
            if isempty(ids)
                ids = 1:numel(obj.val);
            end
            obj.currentCount(ids) = cell(size(obj.currentCount(ids)));
            for fe = {'val', 'diff', 'momentum'}
                f = fe{1};
                for i=ids
                    if force || ~obj.preserve(i)
                        obj.(f){i} = [];
                    end
                end
            end
            if force == 2
                obj.preserve(ids) = logical(obj.preserve(ids).*0);
                obj.ref(ids) = obj.ref(ids).*0;
                obj.lr(ids) = obj.lr(ids).*0;
                obj.decay(ids) = obj.decay(ids).*0;
            end
        end

        function reset(obj)
            obj.val = {};
            obj.diff = {};
            obj.momentum = {};
            obj.method = [];
            obj.currentCount = {};
            obj.maxCount = {};
            obj.ref = [];
            obj.preserve = [];
            obj.lr = [];
            obj.decay = [];
            obj.names = {};
            obj.namesInd = {};
            obj.allInd = [];
        end

        function initVar(obj, id)
            obj.val{id} = [];
            obj.diff{id} = [];
            obj.momentum{id} = [];
            obj.preserve(id) = false;

            obj.currentCount{id} = [];
            obj.maxCount{id} = [];

            obj.ref(id) = int32(0);
            obj.lr(id) = single(0);
            obj.decay(id) = single(0);
            obj.method(id) = obj.methodString.gradient;
        end
        
% ======== SAVE / LOAD ========
        function v = propertyDevice(~)
            v.val = 2;
            v.diff = 2;
            v.momentum = 2;
            v.method = 0;
            v.methodString = 0;
            v.currentCount = 0;
            v.maxCount = 0;
            v.ref = 0;
            v.preserve = 0;

            v.lr = 0;
            v.decay = 0;

            v.names = -1;
            v.namesInd = -1;
            v.allInd = 0;
        end

    end
end
