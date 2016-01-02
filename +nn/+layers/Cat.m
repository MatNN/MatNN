classdef Cat < nn.layers.template.BaseLayer
%CAT Concatenate N data into one data

    properties (SetAccess = protected, Transient)
        default_cat_param = {
                'dim'   3 ...  % HWCN = 1234
            'indices'   {} % empty for normal concatenate operation, each value must be non-identical number
                           % indices specifies each bottom's dim concate to which indices of the whole top
        };
    end

    methods
        function out = f(~, dim, indices, varargin)
            if isempty(indices)
                out   = cat(3, varargin{:});
            else
                D = numel(unique([indices{:}]));
                dims = [1,1,1,1];
                dims0 = size(varargin{1});
                dims(1:numel(dims0)) = dims0;
                dims(dim) = D;
                if opts.gpuMode
                    out = gpuArray.zeros(dims, 'single');
                else
                    out = zeros(dims, 'single');
                end
                switch dim
                    case 1
                        for i=1:numel(indices), out(indices{i},:,:,:) = varargin{i}; end
                    case 2
                        for i=1:numel(indices), out(:,indices{i},:,:) = varargin{i}; end
                    case 3
                        for i=1:numel(indices), out(:,:,indices{i},:) = varargin{i}; end
                    case 4
                        for i=1:numel(indices), out(:,:,:,indices{i}) = varargin{i}; end
                    otherwise
                        error('dim must be 1~4');
                end
            end
        end
        function varargout = b(~, dim, indices, out_diff, varargin)
            varargout = cell(1, numel(varargin));
            if isempty(indices)
                sizeofbtm = cellfun(@(x) size(x, dim), varargin);
                cumuSize = cumsum(sizeofbtm); %[5,3,6] => [5,8,14]
                cumuSize = [0, cumuSize];
                switch dim
                    case 1
                        for i=1:numel(varargin)
                            varargout{i} = out_diff((cumuSize(i)+1):cumuSize(i+1),:,:,:);
                        end
                    case 2
                        for i=1:numel(varargin)
                            varargout{i} = out_diff(:,(cumuSize(i)+1):cumuSize(i+1),:,:);
                        end
                    case 3
                        for i=1:numel(varargin)
                            varargout{i} = out_diff(:,:,(cumuSize(i)+1):cumuSize(i+1),:);
                        end
                    case 4
                        for i=1:numel(varargin)
                            varargout{i} = out_diff(:,:,:,(cumuSize(i)+1):cumuSize(i+1));
                        end
                    otherwise
                        error('dim must be 1~4');
                end
            else
                switch dim
                    case 1
                        for i=1:numel(varargin)
                            varargout{i} = out_diff(indices{i},:,:,:);
                        end
                    case 2
                        for i=1:numel(varargin)
                            varargout{i} = out_diff(:,indices{i},:,:);
                        end
                    case 3
                        for i=1:numel(varargin)
                            varargout{i} = out_diff(:,:,indices{i},:);
                        end
                    case 4
                        for i=1:numel(varargin)
                            varargout{i} = out_diff(:,:,:,indices{i});
                        end
                    otherwise
                        error('dim must be 1~4');
                end
            end
        end
        function forward(obj)
            p = obj.params.slice;
            data = obj.net.data;
            data.val{obj.top} = obj.f(p.dim, p.indices, data.val{obj.bottom});
            data.forwardCount(obj.bottom, obj.top);
        end
        function backward(obj)
            p = obj.params.slice;
            data = obj.net.data;
            bottom_diff = cell(1, numel(obj.bottom));
            [bottom_diff{:}] = obj.b(p.dim, p.indices, data.diff{obj.top}, data.val{obj.bottom});
            data.backwardCount(obj.bottom, obj.top, bottom_diff{:});
        end
        function outSizes = outputSizes(obj, inSizes)
            p = obj.params.cat;
            topSizes = [1, 1, 1, 1];
            topSizes(1:numel(inSizes{1})) = inSizes{1}; % prevent matlab singleton dimension error
            otherDims = setdiff(1:4, p.dim);
            for i=2:numel(inSizes)
                tmpSize = [1, 1, 1, 1];
                tmpSize(1:numel(inSizes{i})) = inSizes{i};
                if isequal(tmpSize(otherDims),  topSizes(otherDims))
                    topSizes(p.dim) = topSizes(p.dim) + tmpSize(p.dim);
                else
                    error('Dimension mismatch.');
                end
            end
            outSizes = {topSizes};
        end
        function setParams(obj)
            obj.setParams@nn.layers.template.BaseLayer();
            p = obj.params.cat;
            assert(numel(p.dim) == 1 && p.dim >= 1 && p.dim <= 4);
        end
        function outSizes = setup(obj, inSizes)
            outSizes = obj.setup@nn.layers.template.BaseLayer(inSizes);
            assert(numel(obj.bottom)>=1);
            assert(numel(obj.top)==1);
            p = obj.params.cat;
            if ~isempty(p.indices)
                assert(numel(p.indices)==numel(obj.bottom));
                assert(all(unique([p.indices{:}])>0));
                sumSize = sum(cell2mat(inSizes'),1);
                assert(numel(unique([p.indices{:}]))==sumSize(3));
            end
        end
    end

end