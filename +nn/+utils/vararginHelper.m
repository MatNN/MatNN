function [ varargout ] = vararginHelper( defaultValues, thePa, varargin )
%VARARGINHELPER Help developers deal with complex inputs of options.
%   USAGE
%   NEWPARAMETERS = VARARGINHELPER( DEFAULTVALUES, PARAMETERS, TRUE )
%   NEWPARAMETERS = VARARGINHELPER( DEFAULTVALUES, PARAMETERS)
%   [EACHPARA,...] = VARARGINHELPER( DEFAULTVALUES, PARAMETERSinCELL, FALSE )
%        
%   INPUT
%   DEFAULTVALUES   A cell or structure stores your input default values
%   PARAMETERS      User specified parameter values
%   last input      An optional boolean value indicates whether your
%                   parameters is a structure or cell. True means structure
%                   , false means cell. Default is true.
%
%   OUTPUT
%   NEWPARAMETERS   A structure. PARAMETERS values replace DEFAULTVLUES
%                   then return this output.
%   EACHPARA,...    Multiple outputs. Each output is the order of 
%                   DEFAULTVALUES option order, and PARAMETERS specified
%                   value of names replace DEFAULTVALUES value, then forms
%                   multiple outputs.
%   
%   EXAMPLE
%   defaultValues = {'K', 100, 'Sigma', 0.8, 'MinSize', 200};
%   userSpecify = {'MinSize', 50, 'Sigma', 1.0};
%   [k, sigma, ms] = vararginHelper(defaultValues, userSpecify, false);
%   then k = 100, sigma = 1.0, ms = 50
%
%   EXAMPLE
%   defaultValues.bins    = 256;
%   defaultValues.spacing = 10;
%   defaultValues.max     = 786;
%   userSpecify.bins      = 128;
%   newValues = vararginHelper(defaultValues, userSpecify);
%   then newValues.bins = 128, newValues.spacing = 10, newValues.max = 786
%
%   INFORMATION
%   Generally, if your input is varargin, then use the cell input form.

if numel(varargin) == 1 && varargin{1} == false
    nameD = defaultValues(1:2:numel(defaultValues));
    valD = defaultValues(2:2:numel(defaultValues));
    nameV = thePa(1:2:numel(thePa));
    valV = thePa(2:2:numel(thePa));

    [allin, reverseOrder] =  ismember(nameV, nameD);
    if any(allin==0)
        error('property settings wrong.');
    end

    res = valD;
    for i=1:numel(reverseOrder)
        res{reverseOrder(i)} = valV{i};
    end
    varargout = res;
else
    if iscell(defaultValues)
        defaultValues = cell2struct(defaultValues(2:2:end), defaultValues(1:2:end),2);
    end
    if iscell(thePa)
        thePa = cell2struct(thePa(2:2:end), thePa(1:2:end),2);
    end
    if isempty(thePa)
        varargout{1} = defaultValues;
        return;
    end
    dF = fieldnames(defaultValues);
    uF = fieldnames(thePa);
    %nF = intersect(uF,dF);
    %for i=1:numel(nF)
    %    defaultValues.(nF{i}) = thePa.(nF{i});
    %end
    for i=uF'
        if isfield(defaultValues, i{1})
            defaultValues.(i{1}) = thePa.(i{1});
        end
    end
    varargout{1} = defaultValues;
end
end