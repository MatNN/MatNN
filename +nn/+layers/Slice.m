classdef Slice < nn.layers.template.BaseLayer
%SLICE Slice a blob into many blobs
%  NOTICE
%  This layer can also extract desired portion of a blob,
%  not just slice it
%
%  EXAMPLE
%  trainer.add({
%      'type'   'nn.layers.slice' ...
%      'name'   'sliceIt'         ...
%      'bottom' 'data'            ...
%      'top'    {'sliceRes1', 'sliceRes2', 'sliceRes3', 'sliceRes4'} ...
%      'slice_param' {
%          'dim' 3 ...
%          'indices' {1:3, 4:6, 7:9, 10}
%          }
%      });

    properties (SetAccess = protected, Transient)
        default_slice_param = {
                'dim' 3 ...  % HWCN = 1234
            'indices' {}
        };
    end

    methods
        function varargout = f(~, dim, indices, in)
            K = numel(indices);
            varargout = cell(1,K);
            switch dim
                case 1
                    for i=1:K
                        varargout{K} = in(indices{i},:,:,:);
                    end
                case 2
                    for i=1:K
                        varargout{K} = in(:,indices{i},:,:);
                    end
                case 3
                    for i=1:K
                        varargout{K} = in(:,:,indices{i},:);
                    end
                case 4
                    for i=1:K
                        varargout{K} = in(:,:,:,indices{i});
                    end
                otherwise
                    for i=1:K
                        varargout{K} = in(:,:,indices{i},:);
                    end
            end
        end
        function in_diff = b(~, dim, indices, in, varargin)
            in_diff = in.*single(0); %works for normal array and gpuArray
            K = numel(indices);
            switch dim
                case 1
                    for i=1:K
                        in_diff(indices{i},:,:,:) = varargin{i};
                    end
                case 2
                    for i=1:K
                        in_diff(:,indices{i},:,:) = varargin{i};
                    end
                case 3
                    for i=1:K
                        in_diff(:,:,indices{i},:) = varargin{i};
                    end
                case 4
                    for i=1:K
                        in_diff(:,:,:,indices{i}) = varargin{i};
                    end
                otherwise
                    for i=1:K
                        in_diff(:,:,indices{i},:) = varargin{i};
                    end
            end
        end
        function forward(obj, nnObj, l, opts, data, net)
            p = obj.params.slice;
            data.val{l.top} = obj.f(p.dim, p.indices, data.val{l.bottom});
        end
        function backward(obj, nnObj, l, opts, data, net)
            p = obj.params.slice;
            nn.utils.accumulateData(opts, data, l, obj.b(p.dim, p.indices, data.val{l.bottom}, data.diff{l.top}));
        end
        function outSizes = outputSizes(obj, opts, l, inSizes, varargin)
            K = numel(obj.params.slice.indices);
            outSizes = cell(1, K);
            d1 = inSizes{1}(1);
            d2 = inSizes{1}(2);
            d3 = inSizes{1}(3);
            d4 = inSizes{1}(4);
            switch obj.params.slice.dim
                case 1
                    for i=1:K
                        outSizes{K} = [numel(obj.params.slice.indices{i}),d2,d3,d4];
                    end
                case 2
                    for i=1:K
                        outSizes{K} = [d1,numel(obj.params.slice.indices{i}),d3,d4];
                    end
                case 3
                    for i=1:K
                        outSizes{K} = [d1,d2,numel(obj.params.slice.indices{i}),d4];
                    end
                case 4
                    for i=1:K
                        outSizes{K} = [d1,d2,d3, numel(obj.params.slice.indices{i})];
                    end
                otherwise
                    for i=1:K
                        outSizes{K} = [d1,d2,numel(obj.params.slice.indices{i}),d4];
                    end
            end
        end
        function [outSizes, resources] = setup(obj, opts, l, inSizes, varargin)
            [outSizes, resources] = obj.setup@nn.layers.template.BaseLayer(opts, l, inSizes, varargin{:});
            assert(numel(l.bottom)==1);
            assert(numel(l.top)==numel(obj.params.slice.indices));
        end
    end

end