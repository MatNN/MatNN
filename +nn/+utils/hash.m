function [ hashString, hashValue, coolMd ] = hash( str, varargin )
%HASH Calculate variable hash value using Java.
%   Usage
%   [ HASHSTRING, HASHVALUE ] = HASH( VARIABLE )
%   [ HASHSTRING, HASHVALUE ] = HASH( VARIABLE, METHOD )
%   [ HASHSTRING, HASHVALUE ] = HASH( VARIABLE, Java MessageDigest object )
%
%   INPUT
%   VARIABLE        A variable to calculated.
%                   Only support class of: numeric, char, cell and struct.
%                   Not supported: table, function_handle and other types.
%   METHOD          Java supported hash methods, like 'MD5', 'SHA-1',
%                   'SHA-256',... etc. Default is 'MD5'.
%
%   OUTPUT
%   HASHSTRING      A String(char) representation of the hash value.
%   HASHVALUE       The hash value in forms of Java BigInteger object.
%   MD              The java security MessageDigest object.
%
%   INFORMATION
%   Empty array, but different types will be encoded to different hash
%   results.
%
%   EXAMPLE
%   import vllab.utils.hash;
%   [hash1, ~] = HASH('Haha!!!', 'SHA-1');
%   [hash2, ~] = HASH({rand(100), struct('f1',[],'f2',3), 4-5i});

import java.security.*;
import java.math.*;

if numel(varargin) == 1
    if ischar(varargin{1})
        md = MessageDigest.getInstance(upper(varargin{1}));
    elseif isjava(varargin{1})
        md = varargin{1};
    else
        error('Second input must be java MessageDigest object or string.');
    end
else
    md = MessageDigest.getInstance('MD5');
end

recursiveUpdate(str, md);
coolMd = md.clone();
hash = md.digest();
bi = BigInteger(1, hash);

hashValue  = bi;
hashString = char(bi.toString(16));

end

function recursiveUpdate(data, md)
    md.update([uint8(class(data)), typecast(size(data), 'uint8')]);
    if isempty(data)
        return;
    end
    if iscell(data)
        for i=1:numel(data)
            recursiveUpdate(data{i}, md);
        end
    elseif isstruct(data)
        fields = fieldnames(data);
        for i=1:numel(fields)
            recursiveUpdate(data.(fields{i}), md);
        end
    elseif isnumeric(data)
        if isreal(data)
            md.update(typecast(data(:), 'uint8'));
        else %if iscomplex
            md.update(typecast(real(data(:)), 'uint8'));
            md.update(typecast(imag(data(:)), 'uint8'));
        end
    elseif ischar(data)
        md.update(uint8(data(:)));
    elseif islogical(data)
        md.update(uint8(data(:)));
    else
        %warning(['Current doesn''t support datatype: ', class(data)]);
        md.update(uint8(char(data)));
    end
end