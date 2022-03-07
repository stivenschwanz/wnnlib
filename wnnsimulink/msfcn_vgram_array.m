function msfcn_vgram_array(block)
% Level-2 MATLAB file S-Function for VGRAM array.

setup(block);
  
%endfunction

function setup(block)
    % Get all parameters
    output_dims = block.DialogPrm(2).Data;
    pattern_length = block.DialogPrm(3).Data;

    % Register the number of ports.
    block.NumInputPorts  = 5;
    block.NumOutputPorts = 1;
    
    % Set up the port properties to be inherited or dynamic.
    block.SetPreCompInpPortInfoToDynamic;
    block.SetPreCompOutPortInfoToDynamic;
    
    % Override the input port properties.
    block.InputPort(1).DatatypeID  = 3;  % uint8
    block.InputPort(1).Complexity  = 'Real';
    block.InputPort(1).Dimensions = pattern_length;

    block.InputPort(2).DatatypeID  = 3;  % uint8
    block.InputPort(2).Complexity  = 'Real';
    block.InputPort(2).Dimensions = pattern_length;

    block.InputPort(3).DatatypeID  = 0; % double
    block.InputPort(3).Complexity  = 'Real';
    block.InputPort(3).Dimensions = output_dims;

    block.InputPort(4).DatatypeID  = 8;  % boolean
    block.InputPort(4).Complexity  = 'Real';
    block.InputPort(4).Dimensions = 1;

    block.InputPort(5).DatatypeID  = 8;  % boolean
    block.InputPort(5).Complexity  = 'Real';
    block.InputPort(5).Dimensions = 1;
    
    % Override the output port properties.
    block.OutputPort(1).DatatypeID  = 0; % double
    block.OutputPort(1).Complexity  = 'Real';
    block.OutputPort(1).Dimensions = output_dims;
    
    % Register the parameters.
    block.NumDialogPrms     = 8;
    block.DialogPrmsTunable = {'Nontunable','Nontunable','Nontunable','Nontunable','Nontunable','Nontunable','Nontunable','Tunable'};
    
    % Register the sample times.
    %  [0 offset]            : Continuous sample time
    %  [positive_num offset] : Discrete sample time
    %
    %  [-1, 0]               : Inherited sample time
    %  [-2, 0]               : Variable sample time
    block.SampleTimes = [-1 0];
    
    % -----------------------------------------------------------------
    % Options
    % -----------------------------------------------------------------
    % Specify if Accelerator should use TLC or call back to the 
    % MATLAB file
    block.SetAccelRunOnTLC(false);
    
    % Specify the block's operating point compliance. The block operating 
    % point is used during the containing model's operating point save/restore)
    % The allowed values are:
    %   'Default' : Same the block's operating point as of a built-in block
    %   'UseEmpty': No data to save/restore in the block operating point
    %   'Custom'  : Has custom methods for operating point save/restore
    %                 (see GetOperatingPoint/SetOperatingPoint below)
    %   'Disallow': Error out when saving or restoring the block operating point.
    block.OperatingPointCompliance = 'Default';
    
    % -----------------------------------------------------------------
    % Register the methods called during update diagram/compilation.
    % -----------------------------------------------------------------
    block.RegBlockMethod('PostPropagationSetup', @DoPostPropSetup);
    block.RegBlockMethod('CheckParameters', @CheckPrms);
    block.RegBlockMethod('Start', @Start);
    block.RegBlockMethod('Outputs', @Outputs);
    block.RegBlockMethod('Terminate', @Terminate);

%endfunction

function DoPostPropSetup(block)
    % Get parameters
    output_dims = block.DialogPrm(2).Data;

    %% Setup Dwork
    block.NumDworks = 2;
  
    block.Dwork(1).Name            = 'codec_ownership';   
    block.Dwork(1).Dimensions      = 1;
    block.Dwork(1).DatatypeID      = 8;  % boolean
    block.Dwork(1).Complexity      = 'Real';
    block.Dwork(1).UsedAsDiscState = true;

    block.Dwork(2).Name            = 'last_output_values';   
    block.Dwork(2).Dimensions      = prod(output_dims);
    block.Dwork(2).DatatypeID      = 0; % double
    block.Dwork(2).Complexity      = 'Real';
    block.Dwork(2).UsedAsDiscState = true;

%endfunction

function CheckPrms(block)
    % Get all parameters
    array_id = block.DialogPrm(1).Data;
    output_dims = block.DialogPrm(2).Data;
    pattern_length = block.DialogPrm(3).Data;
    min_mem_size = block.DialogPrm(4).Data;
    max_mem_size = block.DialogPrm(5).Data;
    min_dist = block.DialogPrm(6).Data;
    max_dist = block.DialogPrm(7).Data;
    debug_flag = block.DialogPrm(8).Data;

    % Check the array identifier
    assert(isa(array_id, 'int32'), 'The array id must be an integer.');
    assert(array_id >= 0, 'The array id must be greater than or equal zero.');

    % Check the output dimensions
    assert(isa(output_dims, 'int32'), 'The output dimensions must be an integer.');
    assert(all(output_dims > 0), 'All output dimensions must be greater than zero.');
    assert(length(output_dims) <= 2, 'There must be up to two output dimensions.');

    % Check the pattern length
    assert(isa(pattern_length, 'int32'), 'The pattern length must be an integer.');
    assert(pattern_length > 0, 'The pattern length must be greater than zero.');

    % Check the memory sizes
    assert(isa(min_mem_size, 'int32'),'The minimum memory size must be an integer.');
    assert(isa(max_mem_size, 'int32'),'The maximum memory size must be an integer.');
    assert(min_mem_size > 0, 'The minimum memory size must be greater than zero.');
    assert(max_mem_size > 0, 'The maximum memory size must be greater than zero.');
    assert(min_mem_size < max_mem_size, 'The minimum memory size must be smaller than the maximum memory size.');

    % Check the distances
    assert(isa(min_dist, 'int32'),'The minimum Hamming distance must be an integer.');
    assert(isa(max_dist, 'int32'),'The maximum Hamming distance must be an integer.');
    assert(min_dist > 0, 'The minimum Hamming distance must be greater than zero.');
    assert(max_dist > 0, 'The maximum Hamming distance must be greater than zero.');
    assert(min_dist < max_dist, 'The minimum Hamming distance must be smaller than the maximum Hamming distance.');

    % Check the pattern length
    assert(isa(debug_flag, 'logical'), 'The debug flag must be a boolean.');
  
%endfunction

function Start(block)
    % Get all parameters    
    array_id = block.DialogPrm(1).Data;
    output_dims = block.DialogPrm(2).Data;
    pattern_length = block.DialogPrm(3).Data;
    min_mem_size = block.DialogPrm(4).Data;
    max_mem_size = block.DialogPrm(5).Data;
    min_dist = block.DialogPrm(6).Data;
    max_dist = block.DialogPrm(7).Data;

    % Create the VGRAM array
    array_ownership = py.wnnlib.vgram_array.create(array_id, output_dims, pattern_length, min_mem_size, max_mem_size, min_dist, max_dist);

    % Default output values
    output_values = zeros(output_dims, 'double');

    % Store state values
    block.Dwork(1).Data = logical(array_ownership);
    block.Dwork(2).Data = output_values(:);
   
%endfunction

function Outputs(block)
    % Get parameters
    array_id = block.DialogPrm(1).Data;
    output_dims = block.DialogPrm(2).Data;

    % Get recall/learn flag
    recall_flag = block.InputPort(4).Data;
    learn_flag = block.InputPort(5).Data;

   if (recall_flag)
        % Get recall pattern
        recall_pattern = block.InputPort(1).Data;

        % Recall output values using the given input pattern
        output_values = double(py.wnnlib.vgram_array.recall(array_id, recall_pattern));
    
        % Set the output values
        block.OutputPort(1).Data = output_values;

        % Save the output values
        block.Dwork(2).Data = output_values(:);
   else
       % Restore the last output values
        block.OutputPort(1).Data = reshape(block.Dwork(2).Data, output_dims);
   end

   if (learn_flag)
        % Get learn pattern
        learn_pattern = block.InputPort(2).Data;

        % Get output steps
        output_steps = block.InputPort(3).Data;

        % Learn the given input pattern
        py.wnnlib.vgram_array.learn(array_id, learn_pattern, output_steps);
   end

    % Get debug flag parameter
    debug_flag = block.DialogPrm(8).Data;

    if(debug_flag)
        % Debug VGRAM array outputs
        py.wnnlib.vgram_array.debug(array_id);
    end
  
%endfunction

function Terminate(block)
    % Get parameters    
    array_id = block.DialogPrm(1).Data;
    
    % Get array ownership
    array_ownership = block.Dwork(1).Data;
    
    % Delete VGRAM array
    if array_ownership
        py.wnnlib.vgram_array.delete(array_id);
    end

%endfunction
 