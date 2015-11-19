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
        function [top, weights, misc] = forward(obj, opts, top, bottom, weights, misc)
            if opts.gpuMode
                top{1} = obj.gf(bottom{1});
            else
                top{1} = obj.f(bottom{1});
            end
        end
        % Backward function for training/testing routines
        function [bottom_diff, weights_diff, misc] = backward(obj, opts, top, bottom, weights, misc, top_diff, weights_diff)
            if opts.gpuMode
                bottom_diff{1} = obj.gb(bottom{1}, top{1}, top_diff{1});
            else
                bottom_diff{1} = obj.b(bottom{1}, top{1}, top_diff{1});
            end
        end

        % Create resources (weight, misc)
        function resources = createResources(obj, opts, inSizes)
            resources = {};
        end
        % Calc Output sizes
        function outSizes = outputSizes(obj, opts, inSizes)
            outSizes = inSizes;
        end
        % Set parameters
        function setParams(obj, baseProperties)
            p = metaclass(obj);
            for va = p.PropertyList'
                if numel(va.Name) > numel('default__param')
                    if strcmp(va.Name(1:8), 'default_') && strcmp(va.Name((end-5):end), '_param')
                        if isfield(baseProperties, va.Name(9:end))
                            wp = nn.utils.vararginHelper(obj.(va.Name), baseProperties.(va.Name(9:end)));
                        else
                            wp = nn.utils.vararginHelper(obj.(va.Name), obj.(va.Name));
                        end
                        obj.params.(va.Name(9:(end-6))) = wp;
                    end
                end
            end
        end
        % Setup function for training/testing routines
        function [outSizes, resources] = setup(obj, opts, baseProperties, inSizes)
            obj.setParams(baseProperties);
            outSizes  = obj.outputSizes(opts, inSizes);
            resources = obj.createResources(opts, inSizes);
            obj.didSetup = true;
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
        function va = moveTo_private(obj, dest, vaName, va)
            if ~ischar(vaName)
                togo = vaName;
            else
                togo = obj.checkProperty(vaName);
            end
            
            if togo==-1
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

        %check if a property can be on CPU(0) or GPU(1) or both(2) or ignore(-1)
        function v = propertyDevice(~)
            v.params   = -1;
            v.didSetup = -1;
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
    end
    

end