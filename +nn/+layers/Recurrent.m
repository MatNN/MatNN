classdef Recurrent < nn.layers.template.BaseLayer

    properties (SetAccess = protected, Transient)
        default_recurrent_param = {
            'bottom' {} ...
            'top' {} ...
            'time' 1 ...
            'initValue' {} ... % eg. {'in2', 'uniform', {'value', 0.5},   'in3', 'gaussian', {'value', 0.1}};
            'passThrough' false ... %pass internal values (initValue variables) to next batch
        };
    end
    properties (SetAccess = protected)
        src;
        out;
        src_empty;
    end

    methods
        function f(~, varargin)
            error('Nested Layer does not support calling f() or b() directly.');
        end
        function b(~, varargin)
            error('Nested Layer does not support calling f() or b() directly.');
        end
        function [data, net] = forward(obj, nnObj, l, opts, data, net)
            face    = obj.subPhase;
            theTime = obj.params.recurrent.time;

            for t = 1:theTime
                for i = 1:numel(srcId)
                    btmSize = dataSizes{srcId(i)};
                    fprintf(', %s = %s[%d:%d]', varargin{1}.data.names{srcId(i)}, varargin{1}.data.names{l.bottom(i)}, 1+(t-1)*btmSize(4), t*btmSize(4));
                end
                [data, net] = nnObj.f(nnObj, data, net, face, opts.subPhase);
            end





            data.val(data.srcId.(face)) = data.val(data.srcId.(opts.name));
            data.val(data.outId.(face)) = data.val(data.outId.(opts.name));
            [data, net] = nnObj.f(nnObj, data, net, face, opts.subPhase);
            error('tmp');
        end
        function [data, net] = backward(obj, nnObj, l, opts, data, net)
            face    = obj.subPhase;
            theTime = obj.params.recurrent.time;
            data.val(data.srcId.(face)) = data.val(data.srcId.(opts.name));
            data.val(data.outId.(face)) = data.val(data.outId.(opts.name));
            [data, net] = nnObj.b(nnObj, data, net, face, opts.subPhase, outDiffs);
        end
        function outSizes = outputSizes(obj, opts, l, inSizes, varargin)
            if ~isfield(l, 'subPhase') || isempty(l.subPhase)
                error(['Recurrent Layer:', l.name, 'doesn''t have any sublayer.']);
            end


            theTime = obj.params.recurrent.time;
            dataSizes = varargin{2};

            % srcId = varargin{1}.data.srcId.(l.subPhase);
            % outId = varargin{1}.data.outId.(l.subPhase);
            % assert(numel(srcId) == numel(l.bottom));
            % assert(numel(outId) == numel(l.top));

            for i = 1:numel(obj.src)
                btmSize = dataSizes{obj.src(i)};
                btmSize(4) = btmSize(4)/theTime;
                dataSizes{srcId(i)} = btmSize;
            end

            for i = 1:numel(srcId)
                btmSize = dataSizes{obj.src(i)};
                btmSize(4) = btmSize(4)/theTime;
                dataSizes{srcId(i)} = btmSize;
            end

            for t = 1:theTime
                fprintf('%sTime = %d', repmat(' ', [1, 2*(varargin{3}+1)]), t);
                for i = 1:numel(srcId)
                    btmSize = dataSizes{srcId(i)};
                    fprintf(', %s = %s[%d:%d]', varargin{1}.data.names{srcId(i)}, varargin{1}.data.names{l.bottom(i)}, 1+(t-1)*btmSize(4), t*btmSize(4));
                end
                fprintf('\n');
                [dataSizes, ~] = varargin{1}.buildPhase(l.subPhase, dataSizes, varargin{3}+1);
            end

            for i = 1:numel(l.top)
                topSize = dataSizes{srcId(i)};
                topSize(4) = topSize(4)*theTime;
                dataSizes{l.top(i)} = topSize;
            end

            outSizes = dataSizes(l.top);
        end
        % Set parameters
        function setParams(obj, l, nnObj)
            obj.setParams@nn.layers.template.BaseLayer(l);
            p = obj.params.recurrent;
            assert(numel(p.bottom) == numel(l.bottom));
            assert(numel(p.top) == numel(l.top));

            %Get all aux names
            auxNames = nnObj.data.names(l.aux);
            assert(all( ismember(p.top, auxNames) ));
            assert(all( ismember(p.bottom, auxNames) ));

            % convert to ind
            btmIDs = [];
            for i=1:numel(p.bottom)
                btmIDs = [btmIDs, nnObj.data.namesInd.(p.bottom{i})]; %#ok
            end
            topIDs = [];
            for i=1:numel(p.top)
                topIDs = [topIDs, nnObj.data.namesInd.(p.top{i})]; %#ok
            end
            obj.src = btmIDs;
            obj.out = topIDs;
            obj.src_empty = setdiff(nnObj.data.srcId.(l.subPhase), btmIDs,'stable');


            obj.params.conv = p;
        end
        function [outSizes, resources] = setup(obj, opts, l, inSizes, varargin) % varargin{1} = nnObj, {2} = sizes, {3} = nested depth
            obj.setParams(l, varargin{1});
            outSizes  = obj.outputSizes(opts, l, inSizes, varargin{:});
            resources = obj.createResources(opts, l, inSizes, varargin{:});
            obj.didSetup = true;
            %assert(numel(l.aux) == numel(l.bottom)+numel(l.top), 'numel(aux) must be equal to numel(bottom)+numel(top).');
        end
    end
    

end