classdef BaseLayer < handle

    % Default parameters
    properties (SetAccess = protected, Transient)
        %defaultParams;
    end

    % intermediate savings (computed values, recomputed every time)
    properties (Access = protected, Transient)
    end

    % variables (not computed every time, eg. once at launch)
    properties (SetAccess = protected, GetAccess = public)
        params;
        didSetup = false;
        MaxThreadsPerBlock = 1024; %this value will be replaced by your GPU configuration.
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
        function [data, net] = forward(obj, nnObj, l, opts, data, net)
            tmp = net.weightsIsMisc(l.weights);
            weightsInd = l.weights(~tmp);
            miscInd = l.weights(tmp);
            if opts.gpuMode
                data.val{l.top(1)} = obj.gf(data.val{l.bottom(1)});
            else
                data.val{l.top(1)} = obj.f(data.val{l.bottom(1)});
            end
        end
        % Backward function for training/testing routines
        function [data, net] = backward(obj, nnObj, l, opts, data, net)
            if opts.gpuMode
                bottom_diff1 = obj.gb(data.val{l.bottom(1)}, data.val{l.top(1)}, data.diff{l.top(1)});
            else
                bottom_diff1 = obj.b(data.val{l.bottom(1)}, data.val{l.top(1)}, data.diff{l.top(1)});
            end

            data = nn.utils.accumulateData(opts, data, bottom_diff1);
        end

        % Create resources (weight, misc)
        function resources = createResources(obj, opts, l, inSizes, varargin)
            resources = {};
        end
        % Calc Output sizes
        function outSizes = outputSizes(obj, opts, l, inSizes, varargin)
            outSizes = inSizes;
        end
        % Set parameters
        function setParams(obj, l)
            p = metaclass(obj);
            for va = p.PropertyList'
                if numel(va.Name) > numel('default__param')
                    if strcmp(va.Name(1:8), 'default_') && strcmp(va.Name((end-5):end), '_param')
                        if isfield(l, va.Name(9:end))
                            wp = nn.utils.vararginHelper(obj.(va.Name), l.(va.Name(9:end)));
                        else
                            wp = nn.utils.vararginHelper(obj.(va.Name), obj.(va.Name));
                        end
                        obj.params.(va.Name(9:(end-6))) = wp;
                    end
                end
            end
        end
        % Setup function for training/testing routines
        function [outSizes, resources] = setup(obj, opts, l, inSizes, varargin) % varargin{1} = nnObj, {2} = sizes, {3} = nested depth
            obj.setParams(l);
            outSizes  = obj.outputSizes(opts, l, inSizes, varargin{:});
            resources = obj.createResources(opts, l, inSizes, varargin{:});
            obj.didSetup = true;
            if opts.gpuMode
                dg = gpuDevice();
                obj.MaxThreadsPerBlock = dg.MaxThreadsPerBlock;
            end
        end

        % Constructor
        function obj = BaseLayer()
            obj.modifyDefaultParams();
        end

        % Move variables/parameters to CPU/GPU
        function varargout = moveTo(obj, dest)
            p = metaclass(obj);
            if nargout == 0
                for va = p.PropertyList'
                    if va.Abstract || va.Transient || va.Dependent
                        continue;
                    else
                        obj.(va.Name) = obj.moveTo_private(dest, va.Name, obj.(va.Name));
                    end
                end
            elseif nargout == 1
                o = struct();
                for va = p.PropertyList'
                    if va.Abstract || va.Transient || va.Dependent
                        continue;
                    else
                        o.(va.Name) = obj.moveTo_private(dest, va.Name, obj.(va.Name));
                    end
                end
                varargout{1} = o;
            else
                error('Too many outputs.');
            end
            
        end
        % Save variables
        function o = save(obj)
            o = struct();
            p = metaclass(obj);
            for va = p.PropertyList'
                if va.Abstract || va.Transient || va.Dependent
                    continue;
                else
                    o.(va.Name) = obj.moveTo_private('cpu', va.Name, obj.(va.Name));
                end
            end
        end
        % Load variables
        function load(obj, o)
            assert(isstruct(o));
            for ff = fieldnames(o)'
                f = ff{1};
                obj.(f) = o.(f);
            end
        end

        %check if a property can be on CPU(0), GPU(1), both(2), ignore(-1), methodName(to build)
        function v = propertyDevice(~)
            v.params   = -1;
            v.didSetup = -1;
            v.MaxThreadsPerBlock = 0;
        end
    end

    methods (Access = protected)
        function modifyDefaultParams(obj)
            % modify superclass' parameters
        end
        function val = checkProperty(obj, t)
            v = obj.propertyDevice();
            if isfield(v, t)
                val = v.(t);
            else
                val = 2;
            end
        end
        function va = moveTo_private(obj, dest, vaName, va)
            if ~ischar(vaName)
                togo = vaName;
            else
                togo = obj.checkProperty(vaName);
            end
            
            if togo==-1
                return;
            elseif ischar(togo)
                va = obj.(togo)(dest);
                return;
            end
            if isnumeric(va)
                % prevent gpuArray(gpuArray(va))
                if isa(va, 'gpuArray') && (togo==0 || togo==2) && strcmpi(dest, 'cpu')
                    va = gather(va);
                elseif ~isa(va, 'gpuArray') && (togo==1 || togo==2) && strcmpi(dest, 'gpu')
                    va = gpuArray(va);
                end
            elseif iscell(va)
                for i=1:numel(va)
                    va{i} = obj.moveTo_private(dest, togo, va{i});
                end
            elseif isstruct(va)
                for ff=fieldnames(va)'
                    f = ff{1};
                    va.(f) = obj.moveTo_private(dest, togo, va.(f));
                end
            end
        end
    end
    

end