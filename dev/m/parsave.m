function parsave (savefile,varargin)
% parsave v1.0.0 (June 2016).
% parsave allows to save variables to a .mat-file while in a parfor loop.
% This is not possible using the regular 'save' command.
%
% SYNTAX:   parsave(FileName,Variable1,Variable2,...)
%
% NOTE: Unlike 'save', do NOT pass the variable names to this function
% (e.g. 'Variable') but instead the variable itself, so without using the
% quotes. An example of correct usage is:
% CORRECT: parsave('file.mat',x,y,z);
%
% This would be INCORRECT: parsave('file.mat','x','y','z'); %Incorrect!
%
%Copyright (c) 2016 Joost H. Weijs
%ENS Lyon, France
%<jhweijs@gmail.com>
for i=1:nargin-1
    %Get name of variable
    name{i}=inputname(i+1);
    
     %Create variable in function scope
    eval([name{i} '=varargin{' num2str(i) '};']); 
end
%Save all the variables, do this by constructing the appropriate command
%and then use eval to run it.
comstring=['save(''' savefile ''''];
for i=1:nargin-1
    comstring=[comstring ',''' name{i} ''''];
end
comstring=[comstring ');'];
eval(comstring);
end