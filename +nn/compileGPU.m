function compileGPU(varargin)
%COMPILEGPU
% Based on nvcc compile method of MatConvNet.
% MatConvNet : http://www.vlfeat.org/matconvnet
%
% NOTICE:
% 1. Need CUDA v7.0/7.5 and CUDNN v4
% 2. Only works on linux environment
%
%
    arch = computer('arch');

    p.cudaRoot = '/usr/local/cuda';
    p.cudaArch = {};
    p.cudnnRoot = '/usr/local/cudnn';
    p.gpuids = 1:gpuDeviceCount;
    p = nn.utils.vararginHelper(p, varargin);

    if isempty(p.cudaArch)
        p.cudaArch = getGPUArch(p.gpuids);
    end
    p.nvcc_path = fullfile(p.cudaRoot, 'bin', 'nvcc');
    switch arch
        case 'maci64'
            p.cudaLibDir = fullfile(p.cudaRoot, 'lib');
            p.cudnnLibDir = fullfile(p.cudnnRoot, 'lib');
        case 'glnxa64'
            p.cudaLibDir = fullfile(p.cudaRoot, 'lib64');
            p.cudnnLibDir = fullfile(p.cudnnRoot, 'lib64');
        otherwise,  error('Unsupported architecture ''%s''.', arch);
    end
    p.cudnnIncDir = fullfile(p.cudnnRoot, 'include');


    p.srcDir = fullfile(fileparts(fileparts(mfilename('fullpath'))), '+cudnn');
    p.srcIncludeDir = fullfile(p.srcDir, 'include');
    p.srcFileDir = fullfile(p.srcDir, 'src');
    p.destDir = p.srcDir;


    nvccflag = {['-I'  p.cudnnIncDir], ...
                ['-I"' fullfile(matlabroot, 'extern', 'include') '"'], ...
                ['-I"' fullfile(matlabroot, 'toolbox','distcomp','gpu','extern','include') '"'], ...
                ['-I'  p.srcIncludeDir], '-Xcompiler', '-fPIC', p.cudaArch{1}};
    linkflag = {'-largeArrayDims', '-lmwblas', ['-L' p.cudaLibDir], ...
                '-lcudart', '-lcublas', '-lmwgpu', ...
                ['-L' p.cudnnLibDir], '-lcudnn'};

    % Compile mex files
    srcfiles = getAllDesiredFiles([], '.cu', 'src');
    compileAndLink(srcfiles, nvccflag, linkflag, p);
    
    % Compile .cu kernels into .ptx files
    compile_cu2ptx(p.nvcc_path, p.cudaArch);
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
    if numel(cudaArch) > 1
        disp('Please choose an architecture to compile: ');
        for i=1:numel(cudaArch)
            fprintf('[%d] %s\n',i,cudaArch{i});
        end
        ind = input('>> ');
        cudaArch = cudaArch(ind);
    end
end

function compile_cu2ptx(nvcc_path, arch)
    list = getAllDesiredFiles([], '.cu', 'private');
    for i=1:numel(list)
        [fileP, name] = fileparts(list{i});
        outputname = fullfile(fileP, [name, '.ptx']);
        cmd = [nvcc_path, ' -ptx ', list{i}, ' -o ', outputname, ' ', arch{:}];
        fprintf('Compiling %s \n', list{i});
        stat = system(cmd);
        if stat
            fprintf('Command %s failed.\n', cmd);
        end
    end
end

function list = getAllDesiredFiles(fPath, extName, specialFolder) % only finds .cu files in 'specialFolder' directory
    if isempty(fPath)
        [fPath,~] = fileparts(fileparts(mfilename('fullpath')));
    end
    [~,currentDir] = fileparts(fPath);
    l = dir(fPath);
    list = {};
    for i=1:numel(l)
        if strcmp(l(i).name, '.') || strcmp(l(i).name, '..')
            continue;
        elseif l(i).isdir
            list = [list, getAllDesiredFiles(fullfile(fPath, l(i).name), extName, specialFolder)];
        elseif numel(l(i).name)>numel(extName) && strcmpi(l(i).name(end-2:end), extName) 
            if isempty(specialFolder) || (~isempty(specialFolder) && strcmp(currentDir, specialFolder))
                list{end+1} = fullfile(fPath, l(i).name);
            end
        end
    end
end

function compileAndLink(srcfiles, nvccflag, linkflag, p)
    % generate object file name (.o)
    outputfileNames = strrep(srcfiles,'.cu','.o');

    % compile .cu into .o
    for i=1:numel(srcfiles)
        nvcc_cmd = sprintf('"%s" -c "%s" %s -o "%s"', ...
                           p.nvcc_path, srcfiles{i}, ...
                           strjoin(nvccflag), outputfileNames{i});
        fprintf('NVCC: %s\n', nvcc_cmd);
        stat = system(nvcc_cmd);
        if stat, fprintf('Command %s failed.\n', nvcc_cmd); end
    end

    % link .o to .mex
    for i=1:numel(outputfileNames)
        mopts = ['-outdir', p.destDir, linkflag, outputfileNames{i}];
        fprintf('MEX LINK: %s\n', strjoin(mopts));
        mex(mopts{:});
        system(['rm ', outputfileNames{i}]);
    end
end