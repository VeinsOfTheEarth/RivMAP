function Ie = hand_clean(I,cmap)

% This function embeds the getkey.m function written by Jos downloaded 
% from the Matlab File Exhange:
% https://www.mathworks.com/matlabcentral/fileexchange/7465-getkey

% This function allows you to hand draw a mask on the currently-active
% plot. Then you specify with a keypress which class you want the selected
% area to become. Options for keypress:
%         0 -        false (for binary images)
%         1 -        water
%         2 -        land
%         3 -        sediment
%         spacebar - undo
%         esc -      quit
% When done cleaning an image, just click somewhere on it and press esc.

mask = zeros(size(I)); % accumulate all single object masks to this one
BW = 11;
while sum(BW(:)) > 10 % less than 10 pixels is considered empty mask
    
    % Call for hand drawing
    h = imfreehand();
    
    xl  = get(gca,'xlim');
    yl = get(gca,'ylim');
    
    % Wait for keypress
    kp = getkey(1);
    
    set(gca,'xlim',xl,'ylim',yl);

    % Which key was pressed
    if kp == 49 % 1       
        maskmultiplier = 1;
    elseif kp == 48 % 0
        maskmultiplier = 6;
    elseif kp == 50 % 2
        maskmultiplier = 2;
    elseif kp == 51 % 3
        maskmultiplier = 3;
    elseif kp == 32 % space bar if mistake
      continue
    end
    
    % Create scaled mask
    BW = createMask(h) * maskmultiplier;
    
    % Add mask to global mask
    mask(BW>0) = maskmultiplier; 
    
    % Check for escape key
    if kp == 27
      break
    end
end

% Place the masking pixels in the image
Ie = double(I);
Ie(mask>0) = mask(mask>0);
Ie(Ie==6) = 0; % have to put the zeros in afterward

% Plot with proper colormap
imshow(Ie)
if max(max(I)) > 1
    colormap(cmap); caxis([1 5])
end

if max(max(Ie)) == 1
    Ie = logical(Ie);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                          getkey                                    %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ch, tim] = getkey(N,nonascii)

% GETKEY - get a keypress
%   CH = GETKEY waits for a single keypress and returns the ASCII code. It
%   accepts all ascii characters, including backspace (8), space (32),
%   enter (13), etc, that can be typed on the keyboard.
%   Non-ascii keys (ctrl, alt, ..) return a NaN. CH is a double.
%
%   CH = GETKEY(N) waits for N keypresses and returns their ASCII codes.
%   GETKEY(1) is the same as GETKEY without arguments.
%
%   GETKEY('non-ascii') or GETKEY(N,'non-ascii') uses non-documented
%   matlab features to return a string describing the key pressed.
%   In this way keys like ctrl, alt, tab etc. can also distinguished.
%   The return is a string (when N = 1) or a cell array of strings.
%
%   [CH,T] = GETKEY(...) also returns the time between the start of the
%   function and each keypress. This is, however, not that accurate.
%
%   This function is kind of a workaround for getch in C. It uses a modal,
%   but non-visible window, which does show up in the taskbar.
%   C-language keywords: KBHIT, KEYPRESS, GETKEY, GETCH
%
%   Examples:
%
%    fprintf('\nPress any key: ') ;
%    ch = getkey ;
%    fprintf('%c\n',ch) ;
%
%    fprintf('\nPress the Ctrl-key within 3 presses: ') ;
%    ch = getkey(3,'non-ascii')
%    if ismemmber('control', ch),
%      fprintf('OK\n') ;
%    else
%      fprintf(' ... wrong keys ...\n') ;
%    end
%
%  See also INPUT, UIWAIT
%           GETKEYWAIT (File Exchange)

% for Matlab 6.5 and higher
% version 2.0 (jun 2012)
% author : Jos van der Geest
% email  : jos@jasen.nl
%
% History
% 1.0 2005 - creation
% 1.1 dec 2006 - modified lay-out and help
% 1.2 apr 2009 - tested for more recent MatLab releases
% 1.3 jan 2012 - modified a few properties, included check is figure still
%            exists (after comment on FEX by Andrew).
% 2.0 jun 2012 - added functionality to accept multiple key presses

t00 = tic ; % start time of this function

% check the input arguments
error(nargchk(0,2,nargin))
switch nargin
    case 0
        nonascii = '' ;
        N = 1 ;
    case 1
        if ischar(N),
            nonascii = N ;
            N = 1 ;
        else
            nonascii = '' ;
        end
end

if numel(N) ~= 1 || ~isnumeric(N) || N < 1 || fix(N) ~= N
    error('N should be a positive integer scalar.') ;
end

% Determine the callback string to use
if strcmpi(nonascii,'non-ascii'),
    % non-ascii characters are accepted
    nonascii = true ;
    callstr = 'set(gcbf,''Userdata'',get(gcbf,''Currentkey'')) ; uiresume ' ;
elseif isempty(nonascii)
    nonascii = false ;
    % only standard ascii characters are accepted
    callstr = 'set(gcbf,''Userdata'',double(get(gcbf,''Currentcharacter''))) ; uiresume ' ;
else
    error('String argument should be the string ''non-ascii''') ;
end

% Set up the figure
% May be the position property  should be individually tweaked to avoid visibility
fh = figure(...
    'name','Press a key', ...
    'keypressfcn',callstr, ...
    'windowstyle','modal',...
    'numbertitle','off', ...
    'position',[0 0  1 1],...
    'userdata','timeout') ;
try
    ch = cell(1,N) ;
    tim = zeros(1,N) ;
    
    % loop to get N keypresses
    for k=1:N
        % Wait for something to happen, usually a key press so uiresume is
        % executed
        uiwait ;
        tim(k) = toc(t00) ; % get the time of the key press
        ch{k} = get(fh,'Userdata') ;  % and the key itself
        if isempty(ch{k}),
            if nonascii
                ch{k} = NaN ;
            else
                ch{k} = '' ;
            end
        end
    end
    if ~nonascii
        ch = [ch{:}] ;
    else
        if N==1
            ch = ch{1} ; % return as a string
        end
        % return as a cell array of strings
    end
 catch
    % Something went wrong, return empty matrices.
    ch = [] ;
    tim = [] ;
end

% clean up the figure, if it still exists
if ishandle(fh)
    delete(fh) ;
end