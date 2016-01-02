classdef BaseLayer < nn.BaseObject

    % Default parameters
    properties (SetAccess = protected, Transient)
        %defaultParams;
    end

    % intermediate savings (computed values, recomputed every time)
    properties (Access = protected, Transient)
    end

    % variables (not computed every time, eg. once at launch)
    properties (SetAccess = {?nn.BaseObject}, GetAccess = public)
        params;
        didSetup = false;
        MaxThreadsPerBlock = 1024; %this value will be replaced by your GPU configuration.
    end

    % layer info
    properties
        origParams;
        name;
        net;
        bottom = [];
        top = [];
    end

    properties (Access = {?nn.nn})
        disableConnectData = false;
    end


    methods
        % CPU Forward
        % This method is designed to be invoked independently
        function out = f(obj, in)

        end
        % CPU Backward
        % This method is designed to be invoked independently
        function in_diff = b(obj, in, out, out_diff)
        end

        % GPU Forward
        % This method is designed to be invoked independently
        function out = gf(obj, varargin)
            out = obj.f(varargin{:});
        end
        % GPU Backward
        % This method is designed to be invoked independently
        function in_diff = gb(obj, varargin)
            in_diff = obj.b(varargin{:});
        end

        % Forward function for training/testing routines
        function forward(obj)
            %tmp = net.weightsIsMisc(l.weights);
            %weightsInd = l.weights(~tmp);
            %miscInd = l.weights(tmp);
            if obj.net.opts.gpu
                obj.net.data.val{obj.top} = obj.gf(obj.net.data.val{obj.bottom});
            else
                obj.net.data.val{obj.top} = obj.f(obj.net.data.val{obj.bottom});
            end
            obj.net.data.forwardCount(obj.bottom, obj.top);
        end
        % Backward function for training/testing routines
        function backward(obj)
            if obj.net.opts.gpu
                bottom_diff1 = obj.gb(obj.net.data.val{obj.bottom}, obj.net.data.val{obj.top}, obj.net.data.diff{obj.top});
            else
                bottom_diff1 = obj.b(obj.net.data.val{obj.bottom}, obj.net.data.val{obj.top}, obj.net.data.diff{obj.top});
            end
            data.backwardCount(obj.bottom,  obj.top, bottom_diff);
            %nn.utils.accumulateDiff(obj.net.data, obj.bottom,  obj.top, bottom_diff1);
        end

        % Create resources (weight, misc)
        function createResources(obj, inSizes)
        end
        % Calc Output sizes
        function outSizes = outputSizes(obj, inSizes)
            outSizes = inSizes;
        end
        % Set parameters
        function setParams(obj)
            p = metaclass(obj);
            for va = p.PropertyList'
                if numel(va.Name) > numel('default__param')
                    if strcmp(va.Name(1:8), 'default_') && strcmp(va.Name((end-5):end), '_param')
                        if isfield(obj.origParams, va.Name(9:end))
                            wp = nn.utils.vararginHelper(obj.(va.Name), obj.origParams.(va.Name(9:end)));
                        else
                            wp = nn.utils.vararginHelper(obj.(va.Name), obj.(va.Name));
                        end
                        obj.params.(va.Name(9:(end-6))) = wp;
                    end
                end
            end
        end
        % Setup function for training/testing routines
        function outSizes = setup(obj, inSizes)
            obj.setParams();
            outSizes  = obj.outputSizes(inSizes);
            obj.createResources(inSizes);
            obj.didSetup = true;
            if obj.net.opts.gpu
                dg = gpuDevice();
                obj.MaxThreadsPerBlock = dg.MaxThreadsPerBlock;
            end
        end
        function release(obj)
            obj.top = [];
            obj.bottom = [];
        end
        function vars = holdVars(obj)
            vars = [obj.bottom, obj.top];
        end

        function set.bottom(obj, dataIDorNames)
            obj.bottom = obj.connectData('bottom', dataIDorNames);
        end
        function set.top(obj, dataIDorNames)
            obj.top = obj.connectData('top', dataIDorNames);
        end

        function v = propertyDevice(~)
            v.origParams = -1;
            v.params   = -1;
            v.didSetup = -1;
            v.name = -1;
            v.bottom = -1;
            v.top = -1;
            v.net = -2;
            v.MaxThreadsPerBlock = 0;
            v.disableConnectData = -2;
        end
        function varargout = moveTo(obj, dest)
            origDisableConnectData = obj.disableConnectData;
            obj.disableConnectData = true;
            if nargout == 1
                varargout{1} = obj.moveTo@nn.BaseObject(dest);
            elseif nargout == 0
                obj.moveTo@nn.BaseObject(dest);
            end
            obj.disableConnectData = origDisableConnectData;
        end

        % Constructor
        function obj = BaseLayer()
            obj.modifyDefaultParams();
        end

    end

    methods (Access = protected)
        function modifyDefaultParams(obj)
            % modify superclass' parameters
        end
        function connectedIDs = connectData(obj, propName, dataIDorNames, varargin)
            data = obj.net.data;
            if ~obj.disableConnectData
                if iscell(dataIDorNames)
                    newNames = dataIDorNames;
                elseif ischar(dataIDorNames)
                    newNames = {dataIDorNames};
                elseif isreal(dataIDorNames)
                    for i=dataIDorNames
                        assert(data.isVar(i), 'Provided data ID is not valid.');
                    end
                    newNames = data.names(dataIDorNames);
                else
                    error('Not a valid data id or name.');
                end

                origNames = data.names(obj.(propName));
                disconnectedVar = origNames(~ismember(origNames, newNames));
                newconnectedVar = newNames(~ismember(newNames, origNames));
                
                % delete ref
                for i = disconnectedVar
                    dName = i{1};
                    data.releaseVar(dName);
                end
                
                % add ref
                newIDs = zeros(size(newNames));
                for i = 1:numel(newconnectedVar)
                    dName = newconnectedVar{i};
                    data.addVar(dName, varargin{:});
                    if data.isVar(dName)
                        data.holdVar(dName);
                    else
                        data.holdVar(dName);
                    end
                    newIDs(i) = data.getIDbyName(dName);
                end
                connectedIDs = newIDs;
            else
                if iscell(dataIDorNames)
                    ids = zeros(size(dataIDorNames));
                    for i=1:numel(dataIDorNames)
                        ids(i) = data.namesInd.(dataIDorNames{i});
                    end
                elseif ischar(dataIDorNames)
                    ids = data.namesInd.(dataIDorNames);
                elseif isreal(dataIDorNames)
                    for i=dataIDorNames
                        assert(data.isVar(data.names{i}), 'Provided data ID is not valid.');
                    end
                    ids = dataIDorNames;
                else
                    error('Not a valid data id or name.');
                end
                connectedIDs = ids;
            end
        end
    end
    

end