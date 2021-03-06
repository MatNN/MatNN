classdef MNIST < nn.layers.template.DataLayer

    properties (SetAccess = protected, Transient, GetAccess=public)
        default_mnist_param = {
            'type' 'train' ... % train / test
        };
    end

    properties (Access = protected, Transient)
        data;
        label;
    end

    methods
        function v = propertyDevice(obj)
            v = obj.propertyDevice@nn.layers.template.DataLayer();
            v.data = 0;
            v.label = 0;
        end

        function [data, label] = process(obj, rawdata, index, usegpu)
            if usegpu
                data = gpuArray(single(rawdata));
                label = gpuArray(single(obj.label(index)));
            else
                data = single(rawdata);
                label = single(obj.label(index));
            end
        end

        % Calc Output sizes
        function outSizes = outputSizes(obj, inSizes)

            outSizes{1} = [28, 28, 1, obj.params.data.batch_size];
            outSizes{2} = [obj.params.data.batch_size, 1, 1, 1];
        end

        function outSizes = setup(obj, inSizes)
            outSizes = obj.setup@nn.layers.template.DataLayer(inSizes);
            assert(numel(obj.bottom)==0, 'MNIST layer does not accept inputs.');
        end


        function varargout = read(obj, imgList, nThreads, isPrefetch)
            if isPrefetch
                return; %not support
            end
            varargout{1} = obj.data(:,:,:,imgList);
        end

        function [list, ind] = next(obj, list, ind)
            if isempty(obj.data)
                [obj.data, obj.label] = obj.readMNISTDataset();
            end
            if ~isempty(ind)
                return;
            end

            p = obj.params.data;
            bs = p.batch_size;
            n4 = size(obj.data,4);
            if obj.pointer == 0
                obj.batchIndices = zeros(1, ceil(n4/bs)*bs);
                if p.shuffle
                    obj.batchIndices(1:n4) = randperm(n4);
                else
                    obj.batchIndices(1:n4) = 1:n4;
                end
                obj.batchIndices = reshape(obj.batchIndices, bs, []);
                obj.pointer = 1;
            end
            ind = obj.batchIndices(:, obj.pointer);
            ind = ind(ind~=0);% last batch may have 0 indices
            list = ind;

            % end process
            if obj.pointer == size(obj.batchIndices,2)
                obj.pointer = 0;
            else
                obj.pointer = obj.pointer+1;
            end
        end

        function [data4D, Label1D] = readMNISTDataset(obj)
            if strcmpi(obj.params.mnist.type, 'train')
                m = memmapfile(fullfile(obj.params.data.src, 'train-images-idx3-ubyte'),'Offset', 16,'Format', {'uint8' [28 28] 'img'});
                imgData = m.Data;
                clearvars m;
                data4D = zeros(28,28,1,numel(imgData), 'uint8');
                for i=1:numel(imgData)
                    data4D(:,:,1,i) = imgData(i).img';
                end
                m = memmapfile(fullfile(obj.params.data.src, 'train-labels-idx1-ubyte'),'Offset', 8,'Format', 'uint8');
                Label1D = m.Data;
            elseif strcmpi(obj.params.mnist.type, 'test')
                m = memmapfile(fullfile(obj.params.data.src, 't10k-images-idx3-ubyte'),'Offset', 16,'Format', {'uint8' [28 28] 'img'});
                imgData = m.Data;
                clearvars m;
                data4D = zeros(28,28,1,numel(imgData), 'uint8');
                for i=1:numel(imgData)
                    data4D(:,:,1,i) = imgData(i).img';
                end
                m = memmapfile(fullfile(obj.params.data.src, 't10k-labels-idx1-ubyte'), 'Offset', 8,'Format', 'uint8');
                Label1D = m.Data;
            end
            clearvars m;
        end
        
    end
end