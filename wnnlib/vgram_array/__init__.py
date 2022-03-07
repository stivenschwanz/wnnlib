from wnnlib.vgram_array import VGRAMArray
from wnnlib.vgram_array import globals


def create(array_id, output_dims, pattern_length, min_mem_size, max_mem_size, min_dist, max_dist):
    """
    Create a new VGRAM array.

    Parameters:
        array_id (int): Unique identifier of the VGRAM array.
        output_dims (int): Number of output dimensions.
        pattern_length (int): Length of the stored patterns.
        min_mem_size (int): Minimum memory size.
        max_mem_size (int): Maximum memory size.
        min_dist (int): Minimum Hamming distance.
        max_dist (int): Maximum Hamming distance.

    Returns:
        (boolean): Ownership flag indicating whether the API caller owns the VGRAM array or not.
    """
    # Check whether the VGRAM array already exists or not. In the former case, the existing array already has an owner.
    if array_id in globals.g_vgram_arrays:
        return False

    # Sanity check: check whether the array already exists and clean up
    globals.g_vgram_arrays[array_id] = VGRAMArray.VGRAMArray(output_dims=tuple(map(int, output_dims)),
                                                             pattern_length=int(pattern_length),
                                                             min_mem_size=int(min_mem_size),
                                                             max_mem_size=int(max_mem_size),
                                                             min_dist=int(min_dist),
                                                             max_dist=int(max_dist))

    # The caller is the owner of the created array. Thus, the caller is supposed to destroy the array.
    return True


def delete(array_id):
    """
    Delete an existing VGRAM array.

    Parameters:
        array_id (int): Unique identifier of the VGRAM array.
    """
    # Check whether the VGRAM array already exists or not. In the former case, the owner can destroy the array.
    if array_id in globals.g_vgram_arrays:
        globals.g_vgram_arrays[array_id].cleanup()
        globals.g_vgram_arrays.pop(array_id)


def get_ids():
    """
    Get the identifiers of the existing VGRAM arrays.

    Returns:
        (int []): Identifiers of the existing VGRAM arrays.
    """
    globals.g_vgram_arrays.keys()


def clear_ids():
    """
    Delete all existing VGRAM arrays.
    """
    globals.g_vgram_arrays.clear()


def debug(array_id):
    """
    Debug the VGRAM array outputs.

    Parameters:
        array_id (int): Unique identifier of the VGRAM array.
    """
    globals.g_vgram_arrays[array_id].debug()


def recall(array_id, input_pattern):
    """
    Recall the VGRAM array for an input pattern.

    Parameters:
        array_id (int): Unique identifier of the VGRAM array.
        input_pattern (bool[]): Input pattern.

    Returns:
        (int[]): Output values.
    """
    return globals.g_vgram_arrays[array_id].recall(input_pattern)


def learn(array_id, input_pattern, output_steps):
    """
    Decode a sparse representation.

    Parameters:
        array_id (int): Unique identifier of the VGRAM array.
        input_pattern (bool[]): Input pattern.
        output_steps (int[]): Output steps.
    """
    globals.g_vgram_arrays[array_id].learn(input_pattern, output_steps)
