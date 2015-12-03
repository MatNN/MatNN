classdef DataLayer < nn.layers.template.BaseLayer

    properties (SetAccess = protected, Transient)
        default_data_param = {
                    'src' []     ... % A string of a file, which contains image paths with label, or a cell variable of image paths and label
            'root_folder' ''     ...
             'batch_size' 1      ...
                   'full' false  ... % make last batch to fill up images
                'shuffle' false  ...
           'read_threads' 2      ...
               'prefetch' true   ...
        };
    end

    % variables (not computed every time, eg. once at launch)
    properties (SetAccess = {?nn.layers.template.BaseLayer}, GetAccess = public)
        dataPaths    = {};
        dataLabel    = [];
        batchIndices = []; % 2-D matrix, each column is the image indices. And column ind is the batch ind.
        pointer      = 0;
        list         = {};
        ind          = [];
    end

    methods
        %check if a property can be on CPU(0) or GPU(1) or both(2) or ignore(-1)
        function v = propertyDevice(obj)
            v = obj.propertyDevice@nn.layers.template.BaseLayer();
            v.dataPaths    = 0;
            v.dataLabel    = 0;
            v.batchIndices = 0;
            v.pointer      = 0;
            v.list         = 0;
            v.ind          = 0;
        end
        % CPU Forward
        function [data, label] = f(obj)
            p = obj.params.data;
            % Get data
            [obj.list, obj.ind] = obj.next(obj.list, obj.ind);
            
            cache = obj.read(obj.list, p.read_threads, false);
            if p.prefetch
                [listNext, indNext] = obj.next({},[]);
                obj.read(listNext, p.read_threads, true);
            end

            [data, label] = obj.process(cache, obj.ind, false);

            if p.prefetch
                obj.list = listNext;
                obj.ind = indNext;
            else
                obj.list = {};
                obj.ind = [];
            end
        end

        function [data, label] = gf(obj)
            p = obj.params.data;
            % Get data
            [obj.list, obj.ind] = obj.next(obj.list, obj.ind);
            
            cache = obj.read(obj.list, p.read_threads, false);
            if p.prefetch
                [listNext, indNext] = obj.next({},[]);
                obj.read(listNext, p.read_threads, true);
            end

            [data, label] = obj.process(cache, obj.ind, true);

            if p.prefetch
                obj.list = listNext;
                obj.ind = indNext;
            else
                obj.list = {};
                obj.ind = [];
            end
        end

        function in_diff = b(obj, in, out, out_diff)
            in_diff = {};
        end

        function [data, label] = process(obj, rawdata)
            error('must implement.');
        end

        function forward(obj, nnObj, l, opts, data, net)
            if opts.gpuMode
                [data.val{l.top(1)}, data.val{l.top(2)}] = obj.gf();
            else
                [data.val{l.top(1)}, data.val{l.top(2)}] = obj.f();
            end
        end

        function backward(obj, nnObj, l, opts, data, net)
            nn.utils.accumulateData(opts, data, l);
        end

        function outSizes = outputSizes(obj, opts, l, inSizes, varargin)
            error('must implement this method.');
        end

        function [list, ind] = next(obj, list, ind)
            if ~isempty(ind)
                return;
            end

            p = obj.params.data;
            bs = p.batch_size;
            if obj.pointer == 0
                obj.batchIndices = zeros(1, ceil(numel(obj.dataPaths)/bs)*bs);
                if p.shuffle
                    obj.batchIndices(1:numel(obj.dataPaths)) = randperm(numel(obj.dataPaths));
                else
                    obj.batchIndices(1:numel(obj.dataPaths)) = 1:numel(obj.dataPaths);
                end
                obj.batchIndices = reshape(obj.batchIndices, bs, []);
                obj.pointer = 1;
            end
            ind = obj.batchIndices(:, obj.pointer);
            ind = ind(ind~=0);% last batch may have 0 indices
            s = numel(ind);

            list = cell(1,s);
            for i = 1:s
                list{i} = [p.root_folder, filesep, obj.dataPaths{ind(i)}];
            end


            % end process
            if obj.pointer == size(obj.batchIndices,2)
                obj.pointer = 0;
            else
                obj.pointer = obj.pointer+1;
            end
        end

        function varargout = read(obj, imgList, nThreads, isPrefetch)
            error('must implement.');

        end

    end
    

end