% This script will turn the GIST descriptors of the Cifar10 dataset into
% sparse codes.

addpath(genpath('~/mnt/matlab/tools/functions'));
show = @(x, y) sprintf('P@500: %f, R@6k: %f', x(500), y(6000));

% load dataset and dictionary
load cifarData;

% center the input data
%trainData = bsxfun(@rdivide, trainData, sqrt(sum(trainData.^2, 2)));
%testData = bsxfun(@rdivide, testData, sqrt(sum(testData.^2, 2)));
meandata = mean([trainData; testData]);
trainData = bsxfun(@minus, trainData, meandata);
testData = bsxfun(@minus, testData, meandata);

trainData = single(trainData);
testData = single(testData);

% datas are labeld between 0~9, change them into 1~10
trainLabels = single(trainLabels+1);
testLabels = single(testLabels+1);

img_per_class = zeros(10, 1, 'single');
for idx = 1:10
  img_per_class(idx) = length(find(trainLabels == idx));
end

niter = 100;

compared_methods = {'L2'; 'lstm-encoder-decoder'};
dimRed = [256];

net = [];
for idxdim = 1:10
  fprintf('Hashing to %d bits\n', dimRed);

  % image retrieval on L2 distance
  %[P, R] =imgRetrieve(trainData, testData, trainLabels, testLabels, img_per_class, 'Euclidean');
  %fprintf('L2, %s\n', show(P, R));

  % image retrieval on L2 distance on lstm-encoder-decoder
  net = nn.netdef.lstm_encoder_decoder(net);
  outblob = nn.netdef.lstm_encoder_decoder_featExtract(net, {'decoder_final_t1', 'decoder_final_t2', 'decoder_final_t3', 'decoder_final_t4'});
  allData = [reshape(outblob{1}, 128, 60000)', reshape(outblob{2}, 128, 60000)', reshape(outblob{3}, 128, 60000)', reshape(outblob{4}, 128, 60000)'];
  allData = single(allData);
  [P, R] =imgRetrieve(allData(1:59000, :), allData(59001:60000, :), trainLabels, testLabels, img_per_class, 'Euclidean');
  fprintf('lstm-encode-decoder, L2, %s\n', show(P, R));

end