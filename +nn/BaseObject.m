classdef BaseObject < handle

    methods
        % Move variables/parameters to CPU/GPU
        function varargout = moveTo(obj, dest)
            assert(nargout<=1, 'Too many outputs.');

            p = metaclass(obj);
            o = obj;

            if nargout==1, o=struct(); end

            for va = p.PropertyList'
                if va.Abstract || va.Transient || va.Dependent
                    continue;
                else
                    [val, togo] = obj.moveTo_private(dest, va.Name);
                    if togo~=-2
                        o.(va.Name) = val;
                    end
                end
            end

            if nargout==1, varargout{1}=o; end
        end
        % Save variables
        function o = save(obj)
            o = obj.moveTo('cpu');
        end
        % Load variables
        function load(obj, o)
            assert(isstruct(o));
            for ff = fieldnames(o)'
                f = ff{1};
                obj.(f) = o.(f);
            end
        end
    end

    methods (Abstract = true)
    v = propertyDevice(obj)
    %check if a property can be on CPU(0), GPU(1), both(2), current(-1), ignore-notSave(-2), methodName(to build)
    % example:
    % v.params = -1
    % v.didSetup = -1
    % v.(propertyName) = 2
    end

    methods (Access = protected)
        function val = checkProperty(obj, t)
            v = obj.propertyDevice();
            if isfield(v, t)
                val = v.(t);
            else
                val = 2;
            end
        end
        function [va, togo] = moveTo_private(obj, dest, vaName, varargin)
            if ~ischar(vaName)
                togo = vaName;
            else
                togo = obj.checkProperty(vaName);
            end

            if togo==-1
                if ~isempty(varargin)
                    va = varargin{1};
                else
                    va = obj.(vaName);
                end
                return;
            elseif togo==-2
                va = [];
                return;
            elseif ischar(togo)
                va = obj.(togo)(dest);
                return;
            end
            if ~isempty(varargin)
                va = varargin{1};
            else
                va = obj.(vaName);
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
