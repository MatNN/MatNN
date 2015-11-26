function compileGPU(varargin)
    p.cudaRoot = '/usr/local/cuda';
    p.cudaArch = {};
    p.gpuids = 1:gpuDeviceCount;
    p = nn.utils.vararginHelper(p, varargin);

    if isempty(p.cudaArch)
        p.cudaArch = getGPUArch(p.gpuids);
    end
    nvcc_path = fullfile(p.cudaRoot, 'bin', 'nvcc');

    % Compile p2p support
    

    % Compile .cu kernels into .ptx files
    compile_cu2ptx(nvcc_path, p.cudaArch);
end

function cudaArch = getGPUArch(varargin)
    gpuids = [varargin{:}];
    if isempty(gpuids)
        gpuids = 1:gpuDeviceCount;
    end
    cudaArch = cell(1,numel(gpuids));
    for i=1:numel(gpuids)
        device = gpuDevice(gpuids(i));
        cc = strrep(device.ComputeCapability, '.', '');
        cudaArch{i} = sprintf('-gencode=arch=compute_%s,code=\\\"sm_%s,compute_%s\\\" ', cc, cc, cc);
    end
    cudaArch = unique(cudaArch);
    disp('Please choose an architecture to compile: ');
    for i=1:numel(cudaArch)
        fprintf('[%d] %s\n',i,cudaArch{i});
    end
    ind = input('>> ');
    cudaArch = cudaArch(ind);
end

function compile_cu2ptx(nvcc_path, arch)
    [f,~] = fileparts(fileparts(mfilename('fullpath')));
    list = getAllCudaFiles(f);
    for i=1:numel(list)
        [fileP, name] = fileparts(list{i});
        outputname = fullfile(fileP, [name, '.ptx']);
        cmd = [nvcc_path, ' -ptx ', list{i}, ' -o ', outputname, ' ', arch{:}];
        stat = system(cmd);
        if stat
            fprintf('Command %s failed.\n', cmd);
        end
    end
end

%nvcc -ptx affine.cu -gencode arch=compute_35,code=sm_35


function list = getAllCudaFiles(fPath) % only finds .cu files in 'private' directories
    l = dir(fPath);
    [~,currentDir] = fileparts(fPath);
    list = {};
    for i=1:numel(l)
        if strcmp(l(i).name, '.') || strcmp(l(i).name, '..')
            continue;
        elseif l(i).isdir
            list = [list, getAllCudaFiles(fullfile(fPath, l(i).name))];
        elseif strcmpi(l(i).name(end-2:end), '.cu') && strcmp(currentDir, 'private')
            list{end+1} = fullfile(fPath, l(i).name);
        end
    end
end