function o = save(networkParameter)
%SAVE Save intermediate blobs into disk

o.name         = 'Save';
o.generateLoss = false;
o.setup        = @setup;
o.forward      = @forward;
o.backward     = @backward;


default_save_param = {
               'path' '~/'     ... % save path
       'variableName' 'result' ... % .mat file variable name
       'clearOnStart' 'false'  ... % this will clear any thing under 'path'
    'processFunction' [] ...
};

    function [resource, topSizes, param] = setup(l, bottomSizes)
        % resource only have .weight
        % if you have other outputs you want to save or share
        % you can set its learning rate to zero to prevent update
        resource = {};

        if isfield(l, 'save_param')
            wp = nn.utils.vararginHelper(default_save_param, l.save_param);
        else
            wp = nn.utils.vararginHelper(default_save_param, default_save_param);
        end

        assert(numel(l.bottom)==2);
        assert(numel(l.top) == 0, '''Save layer'' does not produce outputs.');
        
        assert(size(bottomSizes{1},4)==numel(bottomSizes{2}), 'numel(names) in file list must match input data number.');
        
        topSizes = {};

        if wp.clearOnStart
            if exist(wp.path, 'dir')
                rmdir(wp.path, 's');
            end
        end

        %return updated param
        param.save_param = wp;
    end


    function [top, weights, misc] = forward(opts, l, weights, misc, bottom, top)
        data = gather(bottom{1});
        for i=1:size(data,4)
            filename = fullfile(l.save_param.path, bottom{2}{i});
            [folders,~,~] = fileparts(filename);
            if ~exist(folders, 'dir'), mkdir(folders); end
            if isempty(l.save_param.processFunction)
                res.(l.save_param.variableName) = data(:,:,:,i);
            else
                res.(l.save_param.variableName) = l.save_param.processFunction(data(:,:,:,i));
            end
            save(filename, '-struct', res);
        end
        
    end


    function [bottom_diff, weights_diff, misc] = backward(opts, l, weights, misc, bottom, top, top_diff, weights_diff)
        bottom_diff = {[],[]};
    end

end