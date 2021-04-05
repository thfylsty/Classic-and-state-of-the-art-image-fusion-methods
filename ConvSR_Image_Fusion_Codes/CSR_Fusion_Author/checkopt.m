function checkopt(inopt, dfopt)
  
% checkopt -- Check options structure for unrecognized fields
%
% Usage:
%       checkopt(inopt, dfopt)
%
% Input:
%       inopt         Input options structure
%       dfopt         Default options structure
%
%   
% Author: Brendt Wohlberg <brendt@lanl.gov>  Modified: 2015-06-17
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'License' file distributed with
% the library.

if ~isempty(inopt) && (~isfield(inopt,'NoOptionCheck') || ~inopt.NoOptionCheck),
  ifnc = fieldnames(inopt);
  for k = 1:length(ifnc),
    ifn = ifnc{k};
    if ~strcmp(ifn, 'NoOptionCheck') && ~isfield(dfopt, ifn),
      warning('SPORCO:UnknownOption', 'Unknown option field %s', ifn);
    end
  end
end

return
