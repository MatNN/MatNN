function batchStruct = generate(useGpu, varargin)
%GENERATE Given a set of parameters and gerante a batch structure for you.
%
%  USAGE
%  GENERATE(USEGPU, 'File', list, 'ProcessFunction', @handle, 'BatchSize', N)
%  GENERATE(USEGPU, 'File', list, 'BatchSize', N)
%  this means you write a @handle to load image
%  your handle must has one input to accept a subset of list.
%  'ProcessFunction' indicates that your data needs read and processing.
%  Note: 1. If your @handle return is not in HWCN shape, than you must specify
%           @handle2 to process your data
%        2. If you don't specify @handle , your list will be treated as
%           an jpeg image list, and read it use vl_imreadjpeg without prefetch.
%  
%  GENERATE(USEGPU, 'Name', name, 'File', var, 'ProcessFunction', @handle, 'BatchSize', N, 'Using4D', false)
%  GENERATE(USEGPU, 'Name', name, 'File', var, 'BatchSize', N, 'Using4D', false)
%  Use this if your data(var) is a matrix, and each column is an data instances
%
%  GENERATE(USEGPU, 'Name', name, 'File', var, 'ProcessFunction', @handle, 'BatchSize', N)
%  GENERATE(USEGPU, 'Name', name, 'File', var, 'BatchSize', N)
%  Use this if your data is a 4-D HWCN tensor.
%
%  GENERATE('Attach', batchStruct1, batchStruct2, ...)
%  Use this if you want to fetch multiple data from different source
%  Notice, in order to syncronize the actual index of data, the Random function other than 'batchStruct1' will
%  be discard.
%  :    generate('Attach', dataStruct, labelStruct)
%  and fetch() will generate index using the ProcessFunction of Data, and pass it to the ProcessFunction of label
%  After doing the fetch function call, you will get a struct, each field is the data/label name you defined
%
%
%
%  Other parameters:
%  'Random'         0: don't randomize data
%                   1: if you want each time get a batch of random sampled data
%                   2: random all data each full iter and get batch in order.
%                   @handle: your own random function, you can implement a random algorithm
%                            follows the error rate or get rare data more frequently.
%                            your @handle must accept 4 inputs,
%                            (totalTimesOfDataSampled, lastErrorRateOfData, lastBatchIndices, lastBatchErrors)
%                   Note, only 0 and 2 support epoch training, 1 and @handle do not.
%  'Prefetch'       True/False
%                   If your ProcessFunction using prefetch paradigm, you should specify here.
%                   Prefetch paradigm:
%                       The first call will return an empty data, and starts read data on background.
%                       The second call will return the last readed data.
%                       Repeat above process.
%            
%  NOTICE:
%  Your ProcessFunction should output a 'single' type tensor.
%
%
%
%

if strcmpi(useGpu,'Attach')
    batchStruct = attachProcedure(varargin{:});
    return;
end

defaultValues = {'Name', 'data', 'Prefetch', false, 'File', [], 'ProcessFunction', [], 'BatchSize', 1, 'Random', uint8(0), 'Using4D', true};
[name, prefetch, F, P, N, rnd, fourD] = nn.utils.vararginHelper(defaultValues, varargin, false);

if ~isempty(F)
    if iscell(F)
        m = numel(F);
    else
        if isreal(F) && ~fourD
            m = size(F, 2);
        elseif isreal(F) && fourD
            m = size(F, 4);
        else
            error('Your data must be an cell of file list, or a real number tensor.');
        end
    end
else
    error('Data is empty.');
end
if N == 0
    error('You must specify a legal batch size.');
end
if m == 0
    error('Number == 0.');
end
if N > m
    warning('Your batch size is larger than data number, automatically use the data number as batch size...');
end

batchStruct.F  = {F};
batchStruct.name = {name};
batchStruct.m = m;
batchStruct.N = N;
batchStruct.fourD = fourD;
batchStruct.rnd = rnd;
batchStruct.prefetch = prefetch;
batchStruct.prefetchIndices = [];
if ~isa(rnd, 'function_handle')
    if rnd == 0 || rnd == 2
        batchStruct.batchNumber = ceil(m/N);
    end
end

%for fetch() use
batchStruct.totalTimesOfDataSampled = zeros(1,m,'uint32');
batchStruct.lastErrorRateOfData = ones(1,batchStruct.m, 'single')*realmax('single');
batchStruct.lastBatchErrors = zeros(1,m,'single');
batchStruct.lastBatchIndices = [];
batchStruct.lastIndOfPermute = 0;
if ~isa(rnd, 'function_handle')
    if rnd == 0
        batchStruct.permute = 1:m;
    elseif rnd == 1 || rnd == 2
        batchStruct.permute = randperm(m);
    end
end

if iscell(F)
    batchStruct.fourD = false;
    if isempty(P)
        if useGpu
            batchStruct.Process  = {@defulatImgProcessGPU};
        else
            batchStruct.Process  = {@defulatImgProcess};
            warning('Use default image process procedure, no mean-substrction!!!');
        end
    else
        batchStruct.Process = {P};
    end
else
    if ~fourD
        if isempty(P)
            batchStruct.Process  = {@defulat2DProcess};
        else
            batchStruct.Process = {P};
        end
    else
        batchStruct.Process = {[]};
    end
end

end

function fourDdata = defulat2DProcess(twoDdata)
    % from HN11 to H11N
    [H, N] = size(twoDdata);
    fourDdata = reshape(twoDdata, H, 1, 1, N);
end

function fourDdata = defulatImgProcess(imgList)
    % from HN11 to H11N
    fourDdata = [];
    imgCell = vl_imreadjpeg(imgList, 0);
    for i=1:numel(imgCell)
        im = single(imgCell{i});
        if size(imgCell{i},3) == 1
            fourDdata = cat(4, fourDdata, cat(3, im,im,im));
        else
            fourDdata = cat(4, fourDdata, im);
        end
    end
end
function fourDdata = defulatImgProcessGPU(imgList)
    % from HN11 to H11N
    fourDdata = [];
    imgCell = vl_imreadjpeg(imgList, 0);
    for i=1:numel(imgCell)
        im = single(imgCell{i});
        if size(imgCell{i},3) == 1
            fourDdata = cat(4, fourDdata, cat(3, im,im,im));
        else
            fourDdata = cat(4, fourDdata, im);
        end
    end
    fourDdata = gpuArray(fourDdata);
end

function batchStruct = attachProcedure(varargin)
    names = {};
    prefetch = false;
    for i=1:numel(varargin)
        names = [varargin{i}.name];
        prefetch = prefetch | varargin{i}.prefetch;
    end
    unique_name = unique(names);
    if numel(unique_name) ~= numel(names)
        error('Duplicated struct name!!');
    end
    batchStruct = varargin{1};
    for i = 2:numel(varargin)
        batchStruct.name = [batchStruct.name, varargin{i}.name];
        batchStruct.F = [batchStruct.F, varargin{i}.F];
        batchStruct.Process = [batchStruct.Process, varargin{i}.Process];
        batchStruct.fourD = [batchStruct.fourD, varargin{i}.fourD];
    end
    batchStruct.prefetch = prefetch;
end