function [fourDdata, dataN, batchStruct] = fetch(batchStruct, useGpu, sliceNumber)
%FETCH fetch data for you
%  USAGE
%  [fourDdata, dataN, batchStruct] = fetch(batchStruct)
%
%  dataN is the fourth dimension size of data
%
%  NOTICE 1
%  Each time you call this function you will get a net batchStruct, which stores the current batch informations.
%  So next time you call this function and given the last ouput, you will get the next batch.
%
%  <strong>NOTICE 2<strong>
%  Your custom random procedure must <strong>NOT</strong> generate replicated data indices for each batch.
%  If you want to sample the same data more frequently, you can return a small batch size of data,
%  or raise the probability of the data you want to sampled.
%
%  NOTICE 3
%  Your training procedure must change the value of batchStruct.lastErrorRateOfData if you use your own
%  random procedure.
%
%  NOTICE 4
%  sliceNumber > 1 will slice fourDdata.() into equal number parts.
%  eg. if your original data is fourDdata.data, and fourDdata.label
%      sliceNumber = 3, then the final fourDdata is a cell, size of 3:
%      fourDdata{1}.data, .label; fourDdata{2}.data, .label; fourDdata{3}.data, .label
%  Even sliceNumber = 1 will put the result in a cell
%
%  Example:
%  fileList = {'001.jpg';'002.jpg', ...};
%  dataStruct = nn.batch.generate(true, 'Name', 'data', 'File', fileList, 'BatchSize', 128);
%  labelList = {'001.mat';'002.mat',...};
%  labelStruct = nn.batch.generate(true, 'Name', 'label', 'File', labelList, 'BatchSize', 128);
%
%  batchStruct = nn.batch.generate('Attach', dataStruct, labelStruct);
%  
%
%  for (currentIteration < maxIteration)
%      [res, batchStruct] = nn.batch.fetch(batchStruct, 1);
%      res.data ........
%      res.label ........
%      ........
%      batchStruct.lastErrorRateOfData(batchStruct.lastBatchIndices) = currentErrors;
%  end
%

S = numel(batchStruct.name);

if batchStruct.prefetch
    batchStruct.prefetchIndices = batchStruct.lastBatchIndices;
end

%Generate data indices
if isa(batchStruct.rnd, 'function_handle')
    tp = batchStruct.rnd(batchStruct.totalTimesOfDataSampled, batchStruct.lastErrorRateOfData, batchStruct.lastBatchIndices, batchStruct.lastBatchErrors, batchStruct.N);
    batchStruct.lastBatchIndices = tp;
elseif batchStruct.rnd == 0
    batchStruct.lastBatchIndices = min(batchStruct.lastIndOfPermute+1, batchStruct.m):min(batchStruct.lastIndOfPermute+batchStruct.N, batchStruct.m);
    batchStruct.lastIndOfPermute = batchStruct.lastBatchIndices(end);
elseif batchStruct.rnd == 1
    tmp = randperm(batchStruct.m);
    batchStruct.lastBatchIndices = tmp(1:batchStruct.N);
elseif batchStruct.rnd == 2
    if batchStruct.lastIndOfPermute == batchStruct.m
        batchStruct.permute = randperm(batchStruct.m);
        batchStruct.lastIndOfPermute = 0;
    end
    tmp = min(batchStruct.lastIndOfPermute+1, batchStruct.m):min(batchStruct.lastIndOfPermute+batchStruct.N, batchStruct.m);
    batchStruct.lastBatchIndices = repmat(batchStruct.permute(tmp),S,1);
    batchStruct.lastIndOfPermute = tmp(end);
end

if batchStruct.prefetch && ~isempty(batchStruct.prefetchIndices)
    ind = batchStruct.prefetchIndices;
else
    ind = batchStruct.lastBatchIndices;
end

batchStruct.totalTimesOfDataSampled(ind) = batchStruct.totalTimesOfDataSampled(ind) + 1;

fourDdata = {};
dataN = 0;
for i = 1:S
    tmpData = [];
    if ~batchStruct.fourD(i)
        if iscell(batchStruct.F{i})
            tmpData = batchStruct.Process{i}(batchStruct.F{i}(ind(i,:)));
        else
            tmpData = batchStruct.Process{i}(batchStruct.F{i}(:,ind(i,:)));
        end
    else
        tmpData = batchStruct.F{i}(:,:,:,ind(i,:));
    end

    % slice data
    if ~isempty(tmpData) %if tmpData isempty, must be prefetch
        dataN = size(tmpData, 4);
        subbatchInd = [1:ceil(dataN/sliceNumber):(dataN-1), dataN+1];
        for d = 1:(numel(subbatchInd)-1)
            smalldata = tmpData(:,:,:,subbatchInd(d):subbatchInd(d+1)-1);
            if useGpu >= 1
                fourDdata{d}.(batchStruct.name{i}) = gpuArray(single(smalldata));
            else
                fourDdata{d}.(batchStruct.name{i}) = single(smalldata);
            end
        end
        
    end
end


end