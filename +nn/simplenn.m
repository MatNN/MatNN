function res = simplenn(net, x, dzdy, res, varargin)
%SIMPLENN  Evaluates a simple CNN
%
%  This file is a modified version of original vl_simplenn.m
%  This version reduces the memory usage.
%
%  Forward:
%    'res' is a structure, each field is the tops of layers
%  Backward:
%    'res' is a cell array, each cell is corresponds to a specific layer 
%
%  Details:
%    Weights can be shared, if your weight_name set to the same name.
%    (eg. layers.weights)
%    Tops can be shared. (eg. forward of res)
%    Diffs cannot be shared. (eg. backward of res)
%
%  NOTICE
%  if your layer produces .misc, you need to maintain its gpu/cpu array consistency.
%    
% 
%  NOTICE
%  'res_wrapped' variables are wrapped version of 'res'. And must be wrapped by the carrier class.
%  This function will NOT produces return values because the data of
%  'res_wrapped' will be replaced with a newer 'res'.
%
%  This file is part of the VLFeat library and is made available under
%  the terms of the BSD license (see the COPYING file).

opts.res = [] ;
opts.accumulate = false;
opts.conserveMemory = false ;
opts.sync = false ;
opts.disableDropout = false ;
opts.freezeDropout = false ;
opts.backPropDepth = +inf ;
opts.gpuMode = false;

opts = vl_argparse(opts, varargin);


n = numel(net.layers) ;

if (nargin <= 2) || isempty(dzdy)
  doder = false ;
else
  doder = true ;
end

if nargin <= 3 || isempty(res)
  res.blob  = num2cell(zeros(1, numel(net.blobNames), 'single'));
  res.dzdx  = num2cell(zeros(1, numel(net.blobNames), 'single')); % each cell contains another cell, and the inner cell's length is respected to the number of bottoms that a layer accept
  %res.dzdw  = cell(1, n); % each cell contains another cell, and the inner cell's length is respected to the number of weights of a layer
                          % because weight sharing, so numel(res.dzdw) >= numel(net.weightsNames).
  %res.dzdw  = cell(1, numel(net.weightsNames));  % 好像跟上面不同了。反正weight sharing就是要加上共用此weight的layers的微分結果啊
  res.dzdw  = num2cell(zeros(1, numel(net.weightsNames), 'single')); 
  res.dzdwVisited = false(size(res.dzdw));
  res.time  = zeros(1,n+1);
  res.backwardTime = zeros(1,n+1);

end

for i = fieldnames(x)'
  name2Ind = net.blobNamesIndex.(i{1});
  res.blob{name2Ind} = x.(i{1}); %Because x is a structure, eg. x = struct('data',[],'label',[])
end

for i=1:n
  l = net.layers{i} ;
  forwardBegin = tic ;

  [topBlob, weightUpdate] = net.layerobjs{i}.forward(opts, l, net.weights(l.weights), res.blob(l.bottom));
  %if ~isempty(topBlob)
    res.blob(l.top) = topBlob;
  %else
  %  res.blob(l.top) = {[]};
  %end
  if ~isempty(weightUpdate)
    net.weights(l.weights(weightUpdate{1})) = weightUpdate{2};
  end

  % optionally forget intermediate results
  % 這個可以丟給layer function來做，反正都已經給他們opts了
  forget = opts.conserveMemory ;
  forget = forget & (~doder || strcmp(l.type, 'relu')) ;
  forget = forget & ~net.layerobjs{i}.generateLoss ;
  forget = forget & (~isfield(l, 'rememberOutput') || ~l.rememberOutput) ;
  if forget
      res.blob(l.top) = {[]} ;
  end
  if opts.gpuMode & opts.sync
    % This should make things slower, but on MATLAB 2014a it is necessary
    % for any decent performance.
    wait(gpuDevice) ;
  end
  res.time(i) = toc(forwardBegin) ;
end



if doder

  outputBlob = cellfun(@isempty, net.blobConnectId);
  res.dzdx(outputBlob) = {dzdy} ;

  for i=n:-1:max(1, n-opts.backPropDepth+1)
    l = net.layers{i} ;
    backwardBegin = tic ;

    % 照理說應該對於不同的top的微分應該也要有一個迴圈，但是如果一個layer可以產生兩種output
    % 表示他利用相同的weight來做這件事，所以可以拆成兩個layer function，然後用weight sharing的方式做
    % 如果只是共用部分的weight也可以這樣做，反正共用的weight名稱就一樣，不共用的就不一樣
    % =========重點請看這裡：每個layer只允許一個top，除非：
    %                     此layer不需要backward (backward function只是pass way) , 例如slice?
    
    [tmpdzdx, tmpdzdw] = net.layerobjs{i}.backward(opts, l, net.weights(l.weights), res.blob(l.bottom), res.dzdx(l.top));

    for b = 1:numel(l.bottom)
      if ~isempty(tmpdzdx{b}) 
          if opts.accumulate
              res.dzdx{l.bottom(b)} = res.dzdx{l.bottom(b)} + tmpdzdx{b};
          else
              res.dzdx{l.bottom(b)} = tmpdzdx{b};
          end
          %fprintf('\nb=%d,%s=%.3f,%s=%.3f\n', i,'dzdx max',max(tmpdzdx{b}(:)), 'dzdx min', min(tmpdzdx{b}(:)));
      end
    end
    for w = 1:numel(l.weights) %空的就不加
      if ~isempty(tmpdzdw{w}) 
          if  opts.accumulate || res.dzdwVisited(w)
            res.dzdw{l.weights(w)} = res.dzdw{l.weights(w)} + tmpdzdw{w};
          else
            res.dzdw{l.weights(w)} = tmpdzdw{w};
          end
          res.dzdwVisited(l.weights(w)) = true;
          
          %fprintf('\nlayer %d, w=%d,%s=%.3f,%s=%.3f\n', i, w,'dzdw max',max(tmpdzdw{w}(:)), 'dzdw min', min(tmpdzdw{w}(:)));
      end
    end

    if opts.conserveMemory %delete used dzdx{top} % 這裏不用把loss或accuracy考慮進去，因為loss的微分是1,而accuracy沒有backward
      res.dzdx(l.top) = {[]} ;
    end
    if opts.gpuMode & opts.sync
      wait(gpuDevice) ;
    end
    res.backwardTime(i) = toc(backwardBegin) ;

  end
end


%wrap res again
%%%%no need to do this
