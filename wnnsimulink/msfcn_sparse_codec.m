function msfcn_sparse_codec(block)
% Level-2 MATLAB file S-Function for VGRAM array.

setup(block);
  
%endfunction

function setup(block)
    % Get parameters
    dense_vector_length = block.DialogPrm(2).Data;
    sparse_vector_length = block.DialogPrm(3).Data;
    codec_type = block.DialogPrm(11).Data;
    
    % Set the input/output dimensions and data types properly
    switch codec_type
        case 0 % encoder
            input_dims = dense_vector_length;
            input_dtype = 0;  % double;
            output_dims = sparse_vector_length;
            output_dtype = 3; % uint8
        case 1 % decoder
            input_dims = sparse_vector_length;
            input_dtype = 3; % uint8
            output_dims = dense_vector_length;
            output_dtype = 0;  % double;
        otherwise
            input_dims = [];
            input_dtype = [];
            output_dims = [];
            output_dtype = [];
    end

    % Register the number of ports.
    block.NumInputPorts  = 1;
    block.NumOutputPorts = 1;
    
    % Set up the port properties to be inherited or dynamic.
    block.SetPreCompInpPortInfoToDynamic;
    block.SetPreCompOutPortInfoToDynamic;
    
    % Override the input port properties.
    block.InputPort(1).DatatypeID  = input_dtype;
    block.InputPort(1).Complexity  = 'Real';
    block.InputPort(1).Dimensions = input_dims;
   
    % Override the output port properties.
    block.OutputPort(1).DatatypeID  = output_dtype;
    block.OutputPort(1).Complexity  = 'Real';
    block.OutputPort(1).Dimensions = output_dims;
    
    % Register the parameters.
    block.NumDialogPrms     = 11;
    block.DialogPrmsTunable = {'Nontunable','Nontunable','Nontunable','Nontunable','Nontunable','Nontunable','Nontunable','Nontunable','Nontunable','Tunable','Nontunable'};
    
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
    block.RegBlockMethod('Update', @Update);
    block.RegBlockMethod('Terminate', @Terminate);

%endfunction

function DoPostPropSetup(block)
    % Get parameters
    dense_vector_length = block.DialogPrm(2).Data;
    sparse_vector_length = block.DialogPrm(3).Data;
    codec_type = block.DialogPrm(11).Data;

     % Set the input/output dimensions and data types properly
    switch codec_type
        case 0 % encoder
            output_name =  'sparse_vector_output';
            output_dims = sparse_vector_length;
            output_dtype = 3; % uint8
        case 1 % decoder
            output_name =  'dense_vector_output';
            output_dims = dense_vector_length;
            output_dtype = 0;  % double;
        otherwise
            output_dims = [];
            output_dtype = [];
    end

    %% Setup Dwork
    block.NumDworks = 2;
    block.Dwork(1).Name            = output_name;   
    block.Dwork(1).Dimensions      = output_dims;
    block.Dwork(1).DatatypeID      = output_dtype;
    block.Dwork(1).Complexity      = 'Real';
    block.Dwork(1).UsedAsDiscState = true;

    block.Dwork(2).Name            = 'codec_ownership';   
    block.Dwork(2).Dimensions      = 1;
    block.Dwork(2).DatatypeID      = 8;  % boolean
    block.Dwork(2).Complexity      = 'Real';
    block.Dwork(2).UsedAsDiscState = true;

%endfunction

function CheckPrms(block)
    % Get all parameters
    % codec_id, dense_vector_length, sparse_vector_length, max_depth, learning_rate, min_splitting_volume, min_bounds, max_bounds, sparse_vectors_file, debug_flag
    codec_id = block.DialogPrm(1).Data;
    dense_vector_length = block.DialogPrm(2).Data;
    sparse_vector_length = block.DialogPrm(3).Data;
    max_depth = block.DialogPrm(4).Data;
    learning_rate = block.DialogPrm(5).Data;
    min_splitting_volume = block.DialogPrm(6).Data;
    min_bounds = block.DialogPrm(7).Data;
    max_bounds = block.DialogPrm(8).Data;
    sparse_vectors_file = block.DialogPrm(9).Data;
    debug_flag = block.DialogPrm(10).Data;
    codec_type = block.DialogPrm(11).Data;

    % Check the codec identifier
    assert(isa(codec_id, 'int32'), 'The codec id must be an integer.');
    assert(codec_id >= 0, 'The codec id must be greater than or equal zero.');

    % Check the dense vector length
    assert(isa(dense_vector_length, 'int32'), 'The dense vector length must be an integer.');
    assert(dense_vector_length > 0, 'The dense vector length must be greater than zero.');

    % Check the sparse vector length
    assert(isa(sparse_vector_length, 'int32'), 'The sparse vector length must be an integer.');
    assert(sparse_vector_length > 0, 'The sparse vector length must be greater than zero.');

    % Check the  maximum depth of the kd-tree
    assert(isa(max_depth, 'int32'), 'The maximum depth of the kd-tree must be an integer.');
    assert(max_depth > 0, 'The maximum depth of the kd-tree must be greater than zero.');

    % Check the learning rate
    assert(isa(learning_rate, 'double'), 'The learning rate must be a double.');
    assert(learning_rate >= 0, 'The learning rate must be greater than or equal zero.');
    assert(learning_rate <=1, 'The learning rate must be smaller or equal one.');

    % Check the minimum splitting volume
    assert(isa(min_splitting_volume, 'double'), 'The minimum splitting volume must be a double.');
    assert(min_splitting_volume >= 0, 'The minimum splitting volume must be greater than or equal zero.');

    % Check the bounds
    assert(isa(min_bounds, 'double'),'The minimum bounds must be a double.');
    assert(isa(max_bounds, 'double'),'The maximum bounds must be a double.');
    assert(all(min_bounds < max_bounds), 'The minimum bounds must be smaller than the maximum bounds.');

    % Check the sparse vectors file
    assert(isa(sparse_vectors_file, 'char'),'The sparse vectors file must be a string.');

    % Check the pattern length
    assert(isa(debug_flag, 'logical'), 'The debug flag must be a boolean.');

    % Check the codec type
    assert(isa(codec_type, 'int32'), 'The codec type must be an integer.');
    assert(codec_type >= 0, 'The codec type must be greater than or equal zero.');
    assert(codec_type <= 1, 'The codec type must be smaller than or equal 1.');

%endfunction

function Start(block)
    % Get parameters
    codec_id = block.DialogPrm(1).Data;
    dense_vector_length = block.DialogPrm(2).Data;
    sparse_vector_length = block.DialogPrm(3).Data;
    max_depth = block.DialogPrm(4).Data;
    learning_rate = block.DialogPrm(5).Data;
    min_splitting_volume = block.DialogPrm(6).Data;
    min_bounds = block.DialogPrm(7).Data;
    max_bounds = block.DialogPrm(8).Data;
    sparse_vectors_file = block.DialogPrm(9).Data;
    codec_type = block.DialogPrm(11).Data;

    % Create the sparse codec
    codec_ownership = py.wnnlib.sparse_codec.create(codec_id, max_depth, learning_rate, min_splitting_volume, min_bounds, max_bounds, sparse_vectors_file);

    % Default output values to zero
    switch codec_type
        case 0 % encoder
            output_vector = zeros(1, sparse_vector_length, 'uint8');
        case 1 % decoder
            output_vector = zeros(1, dense_vector_length, 'double');
        otherwise
            output_vector = [];
    end

    % Store output values
    block.Dwork(1).Data = output_vector(:);
    block.Dwork(2).Data = logical(codec_ownership);

%endfunction

function Outputs(block)
    % Get parameters
    codec_id = block.DialogPrm(1).Data;
    debug_flag = block.DialogPrm(10).Data;

    % Restore the last output values
    block.OutputPort(1).Data = block.Dwork(1).Data;

    if(debug_flag)
        % Debug sparse codec
        py.wnnlib.sparse_codec.debug(codec_id, 0);
    end
  
%endfunction

function Update(block)
    % Get parameters
    codec_id = block.DialogPrm(1).Data;
    codec_type = block.DialogPrm(11).Data;

    % Update the output properly
    switch codec_type
        case 0 % encoder
            % Get the dense vector
            input_dense_vector = block.InputPort(1).Data;
            
            % Encode the given dense vector into a sparse vector
            output_sparse_vector = uint8(py.wnnlib.sparse_codec.encode(codec_id, input_dense_vector));
            
            % Save the last output values
            block.Dwork(1).Data = output_sparse_vector(:);
        case 1 % decoder
            % Get the sparse vector
            input_sparse_vector = block.InputPort(1).Data;
            
            % Learn the given input pattern
            output_dense_vector = double(py.wnnlib.sparse_codec.decode(codec_id, input_sparse_vector));
            
            % Save the last output values
            block.Dwork(1).Data = output_dense_vector(:);
        otherwise
            block.Dwork(2).Data = [];
    end
  
%endfunction

function Terminate(block)
    % Get codec parameters    
    codec_id = block.DialogPrm(1).Data;

    % Get codec ownership
    codec_ownership = block.Dwork(2).Data;
    
    % Delete sparse codec
    if codec_ownership
        py.wnnlib.sparse_codec.delete(codec_id);
    end

%endfunction
 