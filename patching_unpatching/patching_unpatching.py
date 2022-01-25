'''
The patching and unpatching are the tool used to cut image to make pathces and combine them
'''
#all the imports
import warnings  # warning import
import numpy as np
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import as_strided


# the input function for the user

def patching_input(start_tensor: torch.tensor , patch_size : int = 256) -> int:
    '''
    the function takes the tensor (channel, width, height) and patch_size
    calculate the optimal padding size.

    Arguments:
        mat: tensor - (channel ,width , height) the image tensor
        patch_size: integer , edge length of the patch_size

    -------------------------------------------------------
    Returns:
        call to _padding_fun and passes the calculated values

    '''
    #checking for the non zero patch size

    if patch_size == 0:
        return "Enter a non zero patch size"

    # changing the tensor to numpy array if not passed
    if isinstance(start_tensor, np.ndarray):
        start_tensor = torch.from_numpy(start_tensor)  # pylint: disable= no-member
    #extract the value for the dimensions from the tensor

    #pylint: disable= W0612
    extra , dim_1 , dim_2 = start_tensor.shape

    #initial value of padding
    padding = 0

    #start a loop

    #calculation for the optimal bleeding size by division
    while True:
        if (dim_1) % (patch_size - 2*padding) == 0 and (dim_2
        ) % (patch_size - 2*padding) == 0:

            #function call to _padding value with calculated values

            #pylint: disable= W0601

            #pylint: disable= C0103

            global calculated_padding  #the values is stored in the variable for later use


            calculated_padding = padding  #storing the value of the calculated padding

            return _padding_fun(initial_tensor= start_tensor,
                    padding=padding,
                    patch_size= patch_size)

        #increase padding
        padding+=1


def _padding_fun(
    initial_tensor: torch.tensor, padding : int =28, patch_size : int= 256
) -> torch.tensor:
    """
    takes the tensor, padding, and patch size input,
    gives the padded tensor and passes to patching function

    Arguments:
        initial_tensor (tensor) - 3 channel or single channel matrix, (channel, width, height)
        padding (integer) -padding value
        patch_size (integer)- side of the square patch
    ---------------------------------------------------------------------------------
    Return:
        calls the patching function, to get patches

    """

    # catching the error, for divisiblity

    #commented for checking
    bleeding = int(padding) #bleeding same as the padding

    # conversion of the tensor to 4d shape for the reflective padding
    initial_tensor_4d = initial_tensor.view(
        1, initial_tensor.shape[0],
        initial_tensor.shape[1],
        initial_tensor.shape[2]
    )

    reflective_padding = nn.ReflectionPad2d(padding)  # initilaizing the reflective padding

    four_d_padded_tensor = reflective_padding(initial_tensor_4d)  # tensor padded

    # conversion for the 3 dimensional array
    three_d_padded_tensor = four_d_padded_tensor.view(
        four_d_padded_tensor.shape[1],
        four_d_padded_tensor.shape[2],
        four_d_padded_tensor.shape[3],
    )

    #passsing the padded tensor to the _patching_fun
    return _patching_fun(
        arr_in=three_d_padded_tensor.numpy(),
        window_shape=patch_size,
        bleeding=bleeding,
        initial_tensor=initial_tensor,
    )



def _patch_fun(arr_in: torch.tensor, window_shape: int, bleeding: int) -> torch.tensor:

    # pylint: disable (too-many-locals)
    """
    the function takes images, patches shape, and bleeding as input and gives the patches as output

    Arguments:
        arr_in (tensor) - 3 channel or single channel matrix, (channel, width, height)
        window_shape (integer)- value for the side length of the square patch
        bleeding (integer)- value for overlap
    -----------------------------------------------------
    Returns:

        patches (tensor)- the patches of the input image, (channel, size_1, size_2 , dim_1, dim_2)

    """

    # calcutalion of the stride with bleeding, taking the bleeding in cosideration
    if (
        window_shape != 2 * bleeding
        and ((arr_in.shape[0] - 2 * bleeding) % (window_shape - 2 * bleeding)) == 0
    ):


        # make the step stride
        step = window_shape - 2 * bleeding
        ndim = arr_in.ndim

        #patching shape
        window_shape = (window_shape,) * ndim

        #step increasing with dimension
        step = (step,) * ndim

        #making the array shape
        arr_shape = np.array(arr_in.shape)

        #patching array build
        window_shape = np.array(window_shape, dtype=arr_shape.dtype)

        # -- build rolling window view
        #storing in the form of tuple
        slices = tuple(slice(None, None, st) for st in step)
        window_strides = np.array(arr_in.strides)

        #using the strides function for patches
        indexing_strides = arr_in[slices].strides

        win_indices_shape = (
            (np.array(arr_in.shape) - np.array(window_shape)) // np.array(step)
        ) + 1

        #formimg the shape of the tensor
        new_shape = tuple(list(win_indices_shape) + list(window_shape))

        #passing the indexing and the window shape to as strided
        strides = tuple(list(indexing_strides) + list(window_strides))

        #storing the value for the return
        arr_out = as_strided(arr_in, shape=new_shape, strides=strides)

        out = arr_out

    return out


# pylint: disable = R1710
def _patching_fun(
    arr_in: torch.tensor, window_shape: int, bleeding: int, initial_tensor=0
) -> torch.tensor:
    """
    The patching functions perform the patch making
    operation for the single and multi-channel tensors

    Arguments:
        arr_in (tensor) - 3 channel or single channel matrix, (channel, width, height)
        window_shape (integer) - side of the square patch
        bleeding (integer)- value for overlap
    --------------------------------------------------

    Returns:
        patches (tensor) - passes to the patch function and combine
        the patches of the input image, (channel, size_1, size_2 , dim_1, dim_2)

    """

    final_mat = [] # a list to store the values

    # checking for three channel matrix
    if arr_in.ndim > 1:
        #loop for the itertation for all the values
        for i in range(arr_in.shape[0]):

            # error catching for the bleeding and patch size
            if not (
                window_shape != 2 * bleeding
                and (
                    (arr_in[i].shape[0] - 2 * bleeding) %
                    (window_shape - 2 * bleeding)
                )
                == 0
            ):
                # pylint: disable = (line-too-long)
                print(
                    "Not correct dimensions, enter the pacth size that divides the original image size minus 2 times of overlap or bleeding"
                )
                # return for the single channel images
                return _padding_fun(initial_tensor)

            #storing values for the three channel tensor
            final_mat.append(
                _patch_fun(arr_in=arr_in[i], window_shape=window_shape, bleeding=bleeding)
            )
        #return for the three channel matrix
        return np.array(final_mat)

# catching the warning from the numpy
warnings.simplefilter(action="ignore", category=FutureWarning)

def _unpatch_fun(
    blown_up_patches: torch.tensor, dimensions: int, blown_up_scale: int
) -> torch.tensor:
    """
    the function takes blown up patches or non blown up patches,
    dimension of the final image, bleeding if present and
    the blown upscale as desired all have to be checked before passing

    Arguments:
        blown_up_patches: a 5 dimensional tensor -
                            (channels, no. of patches along the width of image,
                            no. of patches along the height of the image,
                            the width of the patch, the height of patch )
        dimensions: a tuple- (width, height)
        bleeding: an integer - for overlapping
        blown_up_scale: an integer - the no. of times enlarged
    Returns:
        A tensor of combined patches to form the final image

    """

    # pylint: disable = W0612

     # getting the shape
    channel_dim, dim_1, dim_2, dim_x, dim_y = blown_up_patches.shape

    #conditions for the blown up scale for bleeding calculation
    if blown_up_scale == 1:
        blown_up_bleeding = calculated_padding  #change
    else:
        blown_up_bleeding = blown_up_scale * calculated_padding  #change

    # made the initial matrix with zeros
    initial_array = np.zeros(
        (dim_1, dim_2, dim_x - 2 * blown_up_bleeding, dim_y - 2 * blown_up_bleeding)
    )

    #looping over the pathces and copying the values in bleeding not zero
    if calculated_padding != 0:  #change
        for i in range(0, dim_1):
            for j in range(0, dim_2):
                initial_array[i][j] = blown_up_patches[0][i][j][
                    blown_up_bleeding : dim_x - blown_up_bleeding,
                    blown_up_bleeding : dim_y - blown_up_bleeding,
                ]

    #looping over the pathces and copying the values in bleeding zero
    else:
        for i in range(0, dim_1):
            for j in range(0, dim_2):
                initial_array[i][j] = blown_up_patches[0][i][j]


    #swap axis for alignment fixing
    final_image = initial_array.swapaxes(1, 2).reshape(dimensions)

    #return the final tensor
    return final_image


def unpatching(
    blown_up_patches: torch.tensor, dimensions: int, blown_up_scale: int
) -> torch.tensor:
    """
    the function takes blown up patches or non blown up patches makes the final image,
    this function is used for single and multi-channel images and passing to unpatch
    Arguments:
        blown_up_patches: a 5 dimensional tensor -
                            (channels, no. of patches along the width of image,
                            no. of patches along height of the image,
                            the width of the patch, the height of patch )
        dimensions: a tuple- (width, height)
        blown_up_scale: an integer - the factor for enlargement
    Returns:
        passes to unpatch, get the multichannel patches, and combine them to make final image

    """
    # making matrix for three channel
    final_mat = []
    (
        # pylint: disable = W0612
        channel_dim,
        dim_1,
        dim_2,
        dim_x,
        dim_y,
    ) = blown_up_patches.shape

    #checking the case for the three channel
    if blown_up_patches.shape[0] == 3:

        #looping over the values for the three channel
        for i in range(blown_up_patches.shape[0]):
            blown_up_patches_new = blown_up_patches[i].reshape(
                1, dim_1, dim_2, dim_x, dim_y
            )

            #storing the values for the final return
            final_mat.append(
                #passing to the unpatch function
                _unpatch_fun(
                    blown_up_patches=blown_up_patches_new,
                    dimensions=dimensions,
                    blown_up_scale=blown_up_scale,
                )
            )
    else:

        #for the one channel case
        final_mat.append(
            #passing to the unpatch function
            _unpatch_fun(
                blown_up_patches=blown_up_patches,
                dimensions=dimensions,
                blown_up_scale=blown_up_scale,
            )
        )

    #return the final tensor
    return np.array(final_mat)
