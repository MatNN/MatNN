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
        function [top, weights, misc] = forward(obj, opts, top, bottom, weights, misc)
            p = obj.params.slice;
            top{1} = obj.f(p.dim, p.indices, bottom{1});
        end
        function [bottom_diff, weights_diff, misc] = backward(obj, opts, top, bottom, weights, misc, top_diff, weights_diff)
            p = obj.params.slice;
            bottom_diff{1} = obj.b(p.dim, p.indices, bottom{1}, top_diff{:});
        end
        function outSizes = outputSizes(obj, opts, inSizes)
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
        function [outSizes, resources] = setup(obj, opts, baseProperties, inSizes)
            [outSizes, resources] = obj.setup@nn.layers.template.BaseLayer(opts, baseProperties, inSizes);
            assert(numel(baseProperties.bottom)==1);
            assert(numel(baseProperties.top)==numel(obj.params.slice.indices));
        end
    end

end