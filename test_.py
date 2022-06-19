'''
The testing file for the patching , unpatching function
the image size range from 50 to 500
patch size fixed to 10
padding taken as 0 and 10, types of tensor taken - single channel, three channel
'''
#imports
#pylint: disable= unused-import
import pytest
import numpy as np  
import torch
import torch.nn as nn
import torch.nn.functional as F


from patching_unpatching.patching_unpatching import patching_input, unpatching

#-------------single channel balanced tensor-------------------
mat0 = np.random.rand(1,50,50) #50 dim array

mat1 = np.random.rand(1,100,100) #100 dim array

mat2 = np.random.rand(1,250,250) #200 dim array

mat3 = np.random.rand(1,500,500) #500 dim array

#--------------------------------------------------------------


#-----------------single channel unbalanced tensor--------------------------
mat0_un = np.random.rand(1,50,100) #50 dim array

mat1_un = np.random.rand(1,100,200) #100 dim array

mat2_un = np.random.rand(1,250,500) #200 dim array

mat3_un = np.random.rand(1,500,1000) #500 dim array

#-------------------------------------------------------------------------


#------------------------three channel balanced tensor-------------------------
mat0_3d = np.random.rand(3,50,50) #50 dim array

mat1_3d = np.random.rand(3,100,100) #100 dim array

mat2_3d = np.random.rand(3,250,250) #200 dim array

mat3_3d = np.random.rand(3,500,500) #500 dim array

#---------------------------------------------------------------------------


#-------------three channel balanced tensor--------------------------------------
mat0_3d_un = np.random.rand(3,50,100) #50 dim array

mat1_3d_un = np.random.rand(3,100,200) #100 dim array

mat2_3d_un = np.random.rand(3,250,500) #200 dim array

mat3_3d_un = np.random.rand(3,500,1000) #500 dim array

#---------------------------------------------------------------------------------

#------------------for the single channel tensor--------------------------------

#checking for the balanced  tensor -----------------------------------------------

#checking for the patches with zero padding ----------------------------------

def test_method_0():
    '''
    mat size: 50 X 50
    padding: 0
    patch_size: 10

    return : patch_size (1,5,5,10,10)
    '''
    patch_0 = patching_input(start_tensor =  mat0,
    patch_size = 10)

    assert patch_0.shape == (1, 5, 5, 10, 10)

    return patch_0



def test_method_1():
    '''
    mat size: 100 X 100
    padding: 0
    patch_size: 10

    return : patch_size (1,10,10,10,10)
    '''
    patch_1 = patching_input(start_tensor =  mat1,

    patch_size = 10)

    assert patch_1.shape  == (1, 10, 10, 10, 10)

    return patch_1



def test_method_2():
    '''
    mat size: 250 X 250
    padding: 0
    patch_size: 10

    return : patch_size (1,25,25,10,10)
    '''
    patch_2 = patching_input(start_tensor=  mat2,
    patch_size = 10)

    assert patch_2.shape == (1, 25, 25, 10, 10)

    return patch_2

def test_method_3():
    '''
    mat size: 500 X 500
    padding: 0
    patch_size: 10

    return : patch_size (1,25,25,10,10)
    '''

    patch_3 = patching_input(start_tensor=  mat3,
    patch_size = 10)

    assert patch_3.shape == (1, 50, 50, 10, 10)

    return patch_3

#checking for the patches with 10 padding and 10 overlapping------------------------------------

def test_method_4():
    '''
    mat size: 50 X 50
    padding: 10
    patch_size: 30

    return : patch_size (1,5,5,30,30)
    '''

    patch_4 = patching_input(start_tensor=  mat0,

    patch_size = 30)

    assert patch_4.shape == (1, 5, 5, 30, 30)

    return patch_4



def test_method_5():
    '''
    mat size: 100 X 100
    padding: 5
    patch_size: 30

    return : patch_size (1,10,10,30,30)
    '''
    patch_5 = patching_input(start_tensor= mat1,
    patch_size = 30)

    assert patch_5.shape == (1, 5, 5, 30, 30)

    return patch_5

def test_method_6():
    '''
    mat size: 250 X 250
    padding: 10
    patch_size: 30

    return : patch_size (1,25,25,30,30)
    '''

    patch_6 = patching_input(start_tensor=  mat2,

    patch_size = 30)

    assert patch_6.shape == (1, 25, 25, 30, 30)

    return patch_6

def test_method_7():
    '''
    mat size: 500 X 500
    padding: 10
    patch_size: 30

    return : patch_size (1,25,25,30,30)
    '''

    patch_7 = patching_input(start_tensor=  mat3,

    patch_size = 30)

    assert patch_7.shape == (1, 25, 25, 30, 30)

    return patch_7



#checking for the unbalanced patches in the tensor ----------------------------------

#checking for patches set to zero padding---------------------------------------------------

def test_method_8():
    '''
    mat size: 50 X 100
    padding: 0
    patch_size: 10

    return : patch_size (1,5,10,10,10)
    '''
    patch_8= patching_input(start_tensor=  mat0_un,
    patch_size = 10)

    assert patch_8.shape == (1, 5, 10, 10, 10)

    return patch_8



def test_method_9():
    '''
    mat size: 100 X 200
    padding: 0
    patch_size: 10

    return : patch_size (1,10,20,10,10)
    '''
    patch_9 = patching_input(start_tensor=  mat1_un,
    patch_size = 10)

    assert patch_9.shape == (1, 10, 20, 10, 10)

    return patch_9



def test_method_10():
    '''
    mat size: 250 X 500
    padding: 0
    patch_size: 10

    return : patch_size (1,25,50,10,10)
    '''
    patch_10 = patching_input(start_tensor=  mat2_un,
    patch_size = 10)

    assert patch_10.shape == (1, 25, 50, 10, 10)

    return patch_10



def test_method_11():
    '''
    mat size: 500 X 1000
    padding: 0
    patch_size: 10

    return : patch_size (1,50,100,10,10)
    '''

    patch_11 = patching_input(start_tensor=  mat3_un,

    patch_size = 10)

    assert patch_11.shape == (1, 50, 100, 10, 10)

    return patch_11


#checking for the patches with 10 padding and 10 overlapping------------------------------------

def test_method_12():
    '''
    mat size: 50 X 100
    padding: 10
    patch_size: 30

    return : patch_size (1,5,10,30,30)
    '''

    patch_12 = patching_input(start_tensor=  mat0_un,
    patch_size = 30)

    assert patch_12.shape == (1, 5, 10, 30, 30)

    return patch_12



def test_method_13():
    '''
    mat size: 100 X 200
    padding: 5
    patch_size: 30

    return : patch_size (1,10,20,30,30)
    '''
    patch_13 = patching_input(start_tensor= mat1_un,
    patch_size = 30)

    assert patch_13.shape ==(1, 5, 10, 30, 30)

    return patch_13


def test_method_14():
    '''
    mat size: 250 X 500
    padding: 10
    patch_size: 30

    return : patch_size (1,25,50,30,30)
    '''

    patch_14 = patching_input(start_tensor=  mat2_un,
    patch_size = 30)

    assert patch_14.shape == (1, 25, 50, 30, 30)

    return patch_14



def test_method_15():
    '''
    mat size: 500 X 1000
    padding: 5
    patch_size: 30

    return : patch_size (1,50,100,30,30)
    '''

    patch_15 =  patching_input(start_tensor=  mat3_un,
    patch_size = 30)

    assert patch_15.shape == (1, 25, 50, 30, 30)

    return patch_15

#-----------------------------------------------------------------



#------------------for the three channel tensor--------------------------------

#checking for the balanced  tensor -----------------------------------------------

#checking for the patches with zero padding ----------------------------------



def test_method_16():
    '''
    mat size: 3X50 X 50
    padding: 0
    patch_size: 10

    return : patch_size (3,5,5,10,10)
    '''
    patch_16 = patching_input(start_tensor=  mat0_3d,
    patch_size = 10)

    assert patch_16.shape == (3, 5, 5, 10, 10)

    return patch_16


def test_method_17():
    '''
    mat size: 3X100 X 100
    padding: 0
    patch_size: 10

    return : patch_size (3,10,10,10,10)
    '''
    patch_17 = patching_input(start_tensor=  mat1_3d,
    patch_size = 10)

    assert patch_17.shape == (3, 10, 10, 10, 10)

    return patch_17


def test_method_18():
    '''
    mat size: 3X250 X 250
    padding: 0
    patch_size: 10

    return : patch_size (3,25,25,10,10)
    '''
    patch_18 = patching_input(start_tensor=  mat2_3d,
    patch_size = 10)

    assert patch_18.shape == (3, 25, 25, 10, 10)

    return patch_18


def test_method_19():
    '''
    mat size: 3X500 X 500
    padding: 0
    patch_size: 10

    return : patch_size (3,25,25,10,10)
    '''

    patch_19= patching_input(start_tensor=  mat3_3d,
    patch_size = 10)

    assert patch_19.shape == (3, 50, 50, 10, 10)

    return patch_19


#checking for the patches with 10 padding and 10 overlapping------------------------------------


def test_method_20():
    '''
    mat size: 3X50 X 50
    padding: 10
    patch_size: 30

    return : patch_size (3,5,5,30,30)
    '''

    patch_20 =  patching_input(start_tensor=  mat0_3d,
    patch_size = 30)

    assert patch_20.shape == (3, 5, 5, 30, 30)

    return patch_20


def test_method_21():
    '''
    mat size: 3 X 100 X 100
    padding: 5
    patch_size: 30

    return : patch_size (3,10,10,30,30)
    '''
    patch_21 =  patching_input(start_tensor= mat1_3d,

    patch_size = 30)

    assert patch_21.shape == (3, 5, 5, 30, 30)

    return patch_21


def test_method_22():
    '''
    mat size: 3X250 X 250
    padding: 10
    patch_size: 30

    return : patch_size (3,25,25,30,30)
    '''

    patch_22 =  patching_input(start_tensor=  mat2_3d,
    patch_size = 30)

    assert patch_22.shape == (3, 25, 25, 30, 30)

    return patch_22


def test_method_23():
    '''
    mat size: 3X500 X 500
    padding: 5
    patch_size: 30

    return : patch_size (3,50,50,30,30)
    '''

    patch_23 =  patching_input(start_tensor=  mat3_3d,
    patch_size = 30)

    assert patch_23.shape == (3, 25, 25, 30, 30)

    return patch_23



#checking for the unbalanced patches in the tensor ----------------------------------

#checking for patches set to zero padding---------------------------------------------------


def test_method_24():
    '''
    mat size: 3X50 X 100
    padding: 0
    patch_size: 10

    return : patch_size (3,5,10,10,10)
    '''
    patch_24 =  patching_input(start_tensor=  mat0_3d_un,
    patch_size = 10)

    assert patch_24.shape == (3, 5, 10, 10, 10)

    return patch_24


def test_method_25():
    '''
    mat size: 3X100 X 200
    padding: 0
    patch_size: 10

    return : patch_size (3,10,20,10,10)
    '''
    patch_25 = patching_input(start_tensor=  mat1_3d_un,
    patch_size = 10)

    assert patch_25.shape == (3, 10, 20, 10, 10)

    return patch_25



def test_method_26():
    '''
    mat size: 3X250 X 500
    padding: 0
    patch_size: 10

    return : patch_size (3,25,50,10,10)
    '''
    patch_26 = patching_input(start_tensor=  mat2_3d_un,
    patch_size = 10)

    assert patch_26.shape == (3, 25, 50, 10, 10)

    return patch_26



def test_method_27():
    '''
    mat size: 3X500 X 1000
    padding: 0
    patch_size: 10

    return : patch_size (3,50,100,10,10)
    '''

    patch_27= patching_input(start_tensor=  mat3_3d_un,
    patch_size = 10)

    assert patch_27.shape == (3, 50, 100, 10, 10)

    return patch_27



#checking for the patches with 10 padding and 10 overlapping------------------------------------


def test_method_28():
    '''
    mat size: 3X50 X 100
    padding: 10
    patch_size: 30

    return : patch_size (3,5,10,30,30)
    '''

    patch_28 =  patching_input(start_tensor=  mat0_3d_un,
    patch_size = 30)

    assert patch_28.shape == (3, 5, 10, 30, 30)

    return patch_28



def test_method_29():
    '''
    mat size: 3X100 X 200
    padding: 5
    patch_size: 30

    return : patch_size (3,10,20,30,30)
    '''
    patch_29 =  patching_input(start_tensor= mat1_3d_un,
    patch_size = 30)

    assert patch_29.shape == (3, 5, 10, 30, 30)

    return patch_29


def test_method_30():
    '''
    mat size: 3X250 X 500
    padding: 10
    patch_size: 30

    return : patch_size (3,25,50,30,30)
    '''

    patch_30= patching_input(start_tensor=  mat2_3d_un,
    patch_size = 30)

    assert patch_30.shape == (3, 25, 50, 30, 30)

    return patch_30


def test_method_31():
    '''
    mat size: 3X500 X 1000
    padding: 5
    patch_size: 30

    return : patch_size (3,50,100,30,30)
    '''

    patch_31=patching_input(start_tensor=  mat3_3d_un,
    patch_size = 30)

    assert patch_31.shape == (3, 25, 50, 30, 30)

    return patch_31

#-----------------------------------------------------------------

#testing for the errors---------------------------


def test_method_32():
    '''
    mat size: 50 X 50
    patch_size: 0
    return : error
    '''
    assert patching_input(start_tensor=  mat0,
    patch_size = 0) == "Enter a non zero patch size"



def test_method_33():  #failing zero divisiblity error
    '''
    mat size: 50 X 50
    padding: 0
    patch_size: 30

    return : error
    '''
    assert patching_input(start_tensor=  mat0,
    patch_size = 0) == "Enter a non zero patch size"

#testing for the unpatching ------------------------------

#----------single channel , no padding -------------------


def test_method_40(): #blown up scale is 1
    '''
    mat size: 50 X 50
    padding: 0
    patch_size: 10
    patch_tensor: (1,5,5,10,10)

    blown_up_scale = 1


    return : tensor (1,50,50)
    '''
    unpatch_0 = unpatching(blown_up_patches = test_method_0(),
    dimensions = (50,50),
    blown_up_scale = 1)

    assert np.array_equal(unpatch_0, mat0) == True

    return unpatch_0



def test_method_41(): #blown up scale is 1
    '''
    mat size: 100 X 100
    padding: 0
    patch_size: 10
    patch_tensor: (1,10,10,10,10)

    blown_up_scale = 1


    return : tensor (1,100,100)
    '''
    unpatch_1 = unpatching(blown_up_patches = test_method_1(),
    dimensions = (100,100),
    blown_up_scale = 1)

    assert np.array_equal(unpatch_1, mat1) == True

    return unpatch_1


def test_method_42(): #blown up scale is 1
    '''
    mat size: 250 X 250
    padding: 0
    patch_size: 10
    patch_tensor: (1,25,25,10,10)

    blown_up_scale = 1


    return : tensor (1,250,250)
    '''
    unpatch_2 = unpatching(blown_up_patches = test_method_2(),
    dimensions = (250,250),
    blown_up_scale = 1)

    assert np.array_equal(unpatch_2, mat2) == True

    return unpatch_2



def test_method_43(): #blown up scale is 1
    '''
    mat size: 500 X 500
    padding: 0
    patch_size: 10
    patch_tensor: (1,50,50,10,10)

    blown_up_scale = 1


    return : tensor (1,500,500)
    '''
    unpatch_3 = unpatching(blown_up_patches = test_method_3(),
    dimensions = (500,500),
    blown_up_scale = 1)

    assert np.array_equal(unpatch_3, mat3) == True

    return unpatch_3

#single channel with padding------------------------------------


def test_method_44(): #blown up scale is 1
    '''
    mat size: 50 X 50
    padding: 10
    patch_size: 30
    patch_tensor: (1,5,5,30,30)

    blown_up_scale = 1


    return : tensor (1,50,50)
    '''
    unpatch_4 = unpatching(blown_up_patches = test_method_4(),
    dimensions = (50,50),
    blown_up_scale = 1)

    assert np.array_equal(unpatch_4, mat0) == True

    return unpatch_4


def test_method_45(): #blown up scale is 1
    '''
    mat size: 100 X 100
    padding: 5
    patch_size: 30
    patch_tensor: (1,10,10,30,30)

    blown_up_scale = 1


    return : tensor (1,100,100)
    '''
    unpatch_5 = unpatching(blown_up_patches = test_method_5(),
    dimensions = (100,100),
    blown_up_scale = 1)

    assert np.array_equal(unpatch_5, mat1) == True

    return unpatch_5


def test_method_46(): #blown up scale is 1
    '''
    mat size: 250 X 250
    padding: 10
    patch_size: 30
    patch_tensor: (1,25,25,30,30)

    blown_up_scale = 1


    return : tensor (1,250,250)
    '''
    unpatch_6 = unpatching(blown_up_patches = test_method_6(),
    dimensions = (250,250),
    blown_up_scale = 1)

    assert np.array_equal(unpatch_6, mat2) == True

    return unpatch_6


def test_method_47(): #blown up scale is 1
    '''
    mat size: 500 X 500
    padding: 5
    patch_size: 30
    patch_tensor: (1,50,50,30,30)

    blown_up_scale = 1


    return : tensor (1,500,500)
    '''
    unpatch_7 = unpatching(blown_up_patches = test_method_7(),
    dimensions = (500,500),
    blown_up_scale = 1)

    assert np.array_equal(unpatch_7, mat3) == True

    return unpatch_7



#single channel unbalanced without padding -----------------------

def test_method_48(): #blown up scale is 1
    '''
    mat size: 50 X 100
    padding: 0
    patch_size: 10
    patch_tensor: (1,5,10,10,10)

    blown_up_scale = 1


    return : tensor (1,50,100)
    '''
    unpatch_8 = unpatching(blown_up_patches = test_method_8(),
    dimensions = (50,100),
    blown_up_scale = 1)

    assert np.array_equal(unpatch_8, mat0_un) == True

    return unpatch_8



def test_method_49(): #blown up scale is 1
    '''
    mat size: 100 X 200
    padding: 0
    patch_size: 10
    patch_tensor: (1,10,20,10,10)

    blown_up_scale = 1

    return : tensor (1,100,200)
    '''
    unpatch_9 = unpatching(blown_up_patches = test_method_9(),
    dimensions = (100,200),
    blown_up_scale = 1)

    assert np.array_equal(unpatch_9, mat1_un) == True

    return unpatch_9


def test_method_50(): #blown up scale is 1
    '''
    mat size: 250 X 500
    padding: 0
    patch_size: 10
    patch_tensor: (1,25,50,10,10)

    blown_up_scale = 1


    return : tensor (1,250,500)
    '''
    unpatch_10 = unpatching(blown_up_patches = test_method_10(),
    dimensions = (250,500),
    blown_up_scale = 1)

    assert np.array_equal(unpatch_10, mat2_un) == True

    return unpatch_10


def test_method_51(): #blown up scale is 1
    '''
    mat size: 500 X 1000
    padding: 0
    patch_size: 10
    patch_tensor: (1,50,100,10,10)

    blown_up_scale = 1


    return : tensor (1,500,1000)
    '''
    unpatch_11 = unpatching(blown_up_patches = test_method_11(),
    dimensions = (500,1000),
    blown_up_scale = 1)

    assert np.array_equal(unpatch_11, mat3_un) == True

    return unpatch_11

#checking for the unbalanced tensor with padding -------------------------


def test_method_52(): #blown up scale is 1
    '''
    mat size: 50 X 100
    padding: 10
    patch_size: 30
    patch_tensor: (1,5,10,30,30)

    blown_up_scale = 1


    return : tensor (1,50,100)
    '''
    unpatch_12 = unpatching(blown_up_patches = test_method_12(),
    dimensions = (50,100),
    blown_up_scale = 1)

    assert np.array_equal(unpatch_12, mat0_un) == True

    return unpatch_12


def test_method_53(): #blown up scale is 1
    '''
    mat size: 100 X 200
    padding: 5
    patch_size: 30
    patch_tensor: (1,10,20,30,30)

    blown_up_scale = 1


    return : tensor (1,100,200)
    '''
    unpatch_13 = unpatching(blown_up_patches = test_method_13(),
    dimensions = (100,200),
    blown_up_scale = 1)

    assert np.array_equal(unpatch_13, mat1_un) == True

    return unpatch_13


def test_method_54(): #blown up scale is 1
    '''
    mat size: 250 X 500
    padding: 10
    patch_size: 30
    patch_tensor: (1,25,50,30,30)

    blown_up_scale = 1


    return : tensor (1,250,500)
    '''
    unpatch_14 = unpatching(blown_up_patches = test_method_14(),
    dimensions = (250,500),
    blown_up_scale = 1)

    assert np.array_equal(unpatch_14, mat2_un) == True

    return unpatch_14



def test_method_55(): #blown up scale is 1
    '''
    mat size: 500 X 1000
    padding: 5
    patch_size: 30
    patch_tensor: (1,50,100,30,30)

    blown_up_scale = 1


    return : tensor (1,500,1000)
    '''
    unpatch_15 = unpatching(blown_up_patches = test_method_15(),
    dimensions = (500,1000),
    blown_up_scale = 1)

    assert np.array_equal(unpatch_15, mat3_un) == True

    return unpatch_15



#testing for three channel tensor with zero padding --------------


def test_method_56(): #blown up scale is 1
    '''
    mat size: 3 X 50 X 50
    padding: 0
    patch_size: 10
    patch_tensor: (3,5,5,10,10)

    blown_up_scale = 1


    return : tensor (3,50,50)
    '''
    unpatch_16 = unpatching(blown_up_patches = test_method_16(),
    dimensions = (50,50),
    blown_up_scale = 1)

    assert np.array_equal(unpatch_16, mat0_3d) == True

    return unpatch_16


def test_method_57(): #blown up scale is 1
    '''
    mat size: 3 X 100 X 100
    padding: 0
    patch_size: 10
    patch_tensor: (3,10,10,10,10)

    blown_up_scale = 1


    return : tensor (3,100,100)
    '''
    unpatch_17 = unpatching(blown_up_patches = test_method_17(),
    dimensions = (100,100),
    blown_up_scale = 1)

    assert np.array_equal(unpatch_17, mat1_3d) == True

    return unpatch_17


def test_method_58(): #blown up scale is 1
    '''
    mat size: 3 X 250 X 250
    padding: 0
    patch_size: 10
    patch_tensor: (3,25,25,10,10)

    blown_up_scale = 1


    return : tensor (3,250,250)
    '''
    unpatch_18 = unpatching(blown_up_patches = test_method_18(),
    dimensions = (250,250),
    blown_up_scale = 1)

    assert np.array_equal(unpatch_18, mat2_3d) == True

    return unpatch_18


def test_method_59(): #blown up scale is 1
    '''
    mat size: 3 X 500 X 500
    padding: 0
    patch_size: 10
    patch_tensor: (3,50,50,10,10)

    blown_up_scale = 1


    return : tensor (3,500,500)
    '''
    unpatch_19 = unpatching(blown_up_patches = test_method_19(),
    dimensions = (500,500),
    blown_up_scale = 1)

    assert np.array_equal(unpatch_19, mat3_3d) == True

    return unpatch_19


#testing for three channel tensor with padding --------------


def test_method_60(): #blown up scale is 1
    '''
    mat size: 3 X 50 X 50
    padding: 10
    patch_size: 20
    patch_tensor: (3,5,5,30,30)

    blown_up_scale = 1


    return : tensor (3,50,50)
    '''
    unpatch_20 = unpatching(blown_up_patches = test_method_20(),
    dimensions = (50,50),
    blown_up_scale = 1)

    assert np.array_equal(unpatch_20, mat0_3d) == True

    return unpatch_20


def test_method_61(): #blown up scale is 1
    '''
    mat size: 3 X 100 X 100
    padding: 5
    patch_size: 30
    patch_tensor: (3,10,10,30,30)

    blown_up_scale = 1


    return : tensor (3,100,100)
    '''
    unpatch_21 = unpatching(blown_up_patches = test_method_21(),
    dimensions = (100,100),
    blown_up_scale = 1)

    assert np.array_equal(unpatch_21, mat1_3d) == True

    return unpatch_21



def test_method_62(): #blown up scale is 1
    '''
    mat size: 3 X 250 X 250
    padding: 10
    patch_size: 30
    patch_tensor: (3,25,25,10,10)

    blown_up_scale = 1


    return : tensor (3,250,250)
    '''
    unpatch_22 = unpatching(blown_up_patches = test_method_22(),
    dimensions = (250,250),
    blown_up_scale = 1)

    assert np.array_equal(unpatch_22, mat2_3d) == True

    return unpatch_22



def test_method_63(): #blown up scale is 1
    '''
    mat size: 3 X 500 X 500
    padding: 5
    patch_size: 30
    patch_tensor: (3,50,50,30,30)

    blown_up_scale = 1


    return : tensor (3,500,500)
    '''
    unpatch_23 = unpatching(blown_up_patches = test_method_23(),
    dimensions = (500,500),
    blown_up_scale = 1)

    assert np.array_equal(unpatch_23, mat3_3d) == True

    return unpatch_23

# unbalanced three channel tensor with zero padding--------------



def test_method_64(): #blown up scale is 1
    '''
    mat size: 3 X 50 X 100
    padding: 0
    patch_size: 10
    patch_tensor: (3,5,10,10,10)

    blown_up_scale = 1


    return : tensor (3,50,100)
    '''
    unpatch_24 = unpatching(blown_up_patches = test_method_24(),
    dimensions = (50,100),
    blown_up_scale = 1)

    assert np.array_equal(unpatch_24, mat0_3d_un) == True

    return unpatch_24


def test_method_65(): #blown up scale is 1
    '''
    mat size: 3 X 100 X 200
    padding: 0
    patch_size: 10
    patch_tensor: (3,10,20,10,10)

    blown_up_scale = 1


    return : tensor (3,100,200)
    '''
    unpatch_25 = unpatching(blown_up_patches = test_method_25(),
    dimensions = (100,200),
    blown_up_scale = 1)

    assert np.array_equal(unpatch_25, mat1_3d_un) == True

    return unpatch_25


def test_method_66(): #blown up scale is 1
    '''
    mat size: 3 X 250 X 550
    padding: 0
    patch_size: 10
    patch_tensor: (3,25,50,10,10)

    blown_up_scale = 1


    return : tensor (3,250,500)
    '''
    unpatch_26 = unpatching(blown_up_patches = test_method_26(),
    dimensions = (250,500),
    blown_up_scale = 1)

    assert np.array_equal(unpatch_26, mat2_3d_un) == True

    return unpatch_26


def test_method_67(): #blown up scale is 1
    '''
    mat size: 3 X 500 X 1000
    padding: 0
    patch_size: 10
    patch_tensor: (3,50,500,10,10)

    blown_up_scale = 1


    return : tensor (3,500,500)
    '''
    unpatch_27 = unpatching(blown_up_patches = test_method_27(),
    dimensions = (500,1000),
    blown_up_scale = 1)

    assert np.array_equal(unpatch_27, mat3_3d_un) == True

    return unpatch_27

# unbalanced three channel tensor with padding--------------


def test_method_68(): #blown up scale is 1
    '''
    mat size: 3 X 50 X 100
    padding: 10
    patch_size: 30
    patch_tensor: (3,5,10,30,30)

    blown_up_scale = 1


    return : tensor (3,50,100)
    '''
    unpatch_28 = unpatching(blown_up_patches = test_method_28(),
    dimensions = (50,100),
    blown_up_scale = 1)

    assert np.array_equal(unpatch_28, mat0_3d_un) == True

    return unpatch_28


def test_method_69(): #blown up scale is 1
    '''
    mat size: 3 X 100 X 200
    padding: 5
    patch_size: 30
    patch_tensor: (3,10,20,30,30)

    blown_up_scale = 1


    return : tensor (3,100,200)
    '''
    unpatch_29 = unpatching(blown_up_patches = test_method_29(),
    dimensions = (100,200),
    blown_up_scale = 1)

    assert np.array_equal(unpatch_29, mat1_3d_un) == True

    return unpatch_29


def test_method_70(): #blown up scale is 1
    '''
    mat size: 3 X 250 X 500
    padding: 10
    patch_size: 30
    patch_tensor: (3,25,50,10,10)

    blown_up_scale = 1


    return : tensor (3,250,500)
    '''
    unpatch_30 = unpatching(blown_up_patches = test_method_30(),
    dimensions = (250,500),
    blown_up_scale = 1)

    assert np.array_equal(unpatch_30, mat2_3d_un) == True

    return unpatch_30


def test_method_71(): #blown up scale is 1
    '''
    mat size: 3 X 500 X 1000
    padding: 5
    patch_size: 30
    patch_tensor: (3,50,100,30,30)

    blown_up_scale = 1


    return : tensor (3,500,1000)
    '''
    unpatch_71 = unpatching(blown_up_patches = test_method_31(),
    dimensions = (500,1000),
    blown_up_scale = 1)

    assert np.array_equal(unpatch_71, mat3_3d_un) == True

    return unpatch_71

#testing for the blown up scale to 4 --------------------------------

def blown_up_function(matrix):
    '''
    The function takes the patches or 3 dim matrix, and returns the blown up patches or matrix

    Arguments:
    patches - either single channel or three channel, dim (channel , length, width, length of patch, width of patch)

    ------------------------------------------------------
    return :

    blown up patches - the patches are enlarged by 4 times , using nearest algo
    '''

    if matrix.ndim > 2:
        if matrix.ndim == 3:
            extra ,dim_1 , dim_2 = matrix.shape
            matrix = matrix.reshape(1,1,dim_1,dim_2)
        else:
            extra , size_1, size_2, dim_1, dim_2 = matrix.shape
            matrix = matrix.reshape(size_1,size_2, dim_1, dim_2)

    matrix = torch.tensor(matrix)
    scale_factor = 4
    scaling = nn.Upsample(scale_factor = scale_factor, mode = 'nearest')

    blown_up = scaling(matrix)

    return blown_up.numpy()


#----------single channel with zero padding blown up scale 4-------------



def test_method_72():
    '''
    mat size: 50 X 50
    padding: 0
    patch_size: 10
    patch_tensor: (1,5,5,10,10)

    blown_up_scale = 4

    return : tensor (200,200)

    '''

    blown_up_mat = blown_up_function(matrix= mat0)
    blown_up_patches = blown_up_function(matrix= test_method_0())

    size_1, size_2 , dim_1 , dim_2 = blown_up_patches.shape

    blown_up_patches = blown_up_patches.reshape(1,size_1,size_2, dim_1,dim_2)

    final_mat = unpatching(blown_up_patches = blown_up_patches,
    dimensions = (200,200),
    blown_up_scale = 4)

    assert (np.abs(blown_up_mat - final_mat)).sum()/ size_1*size_2 == 0.0


def test_method_73():
    '''
    mat size: 100 X 100
    padding: 0
    patch_size: 10
    patch_tensor: (1,10,10,10,10)

    blown_up_scale = 4

    return : tensor (400,400)

    '''

    blown_up_mat = blown_up_function(matrix= mat1)
    blown_up_patches = blown_up_function(matrix= test_method_1())

    size_1, size_2 , dim_1 , dim_2 = blown_up_patches.shape

    blown_up_patches = blown_up_patches.reshape(1,size_1,size_2, dim_1,dim_2)

    final_mat = unpatching(blown_up_patches = blown_up_patches,
    dimensions = (400,400),
    blown_up_scale = 4)

    assert (np.abs(blown_up_mat - final_mat)).sum()/ size_1*size_2 == 0.0



def test_method_74():
    '''
    mat size: 250 X 250
    padding: 0
    patch_size: 10
    patch_tensor: (1,25,25,10,10)

    blown_up_scale = 4

    return : tensor (1000,1000)

    '''

    blown_up_mat = blown_up_function(matrix= mat2)
    blown_up_patches = blown_up_function(matrix= test_method_2())

    size_1, size_2 , dim_1 , dim_2 = blown_up_patches.shape

    blown_up_patches = blown_up_patches.reshape(1,size_1,size_2, dim_1,dim_2)

    final_mat = unpatching(blown_up_patches = blown_up_patches,
    dimensions = (1000,1000),
    blown_up_scale = 4)

    assert (np.abs(blown_up_mat - final_mat)).sum()/ size_1*size_2 == 0.0




def test_method_75():
    '''
    mat size: 500 X 500
    padding: 0
    patch_size: 10
    patch_tensor: (1,50,50,10,10)

    blown_up_scale = 4

    return : tensor (2000,2000)

    '''

    blown_up_mat = blown_up_function(matrix= mat3)
    blown_up_patches = blown_up_function(matrix= test_method_3())

    size_1, size_2 , dim_1 , dim_2 = blown_up_patches.shape

    blown_up_patches = blown_up_patches.reshape(1,size_1,size_2, dim_1,dim_2)

    final_mat = unpatching(blown_up_patches = blown_up_patches,
    dimensions = (2000,2000),
    blown_up_scale = 4)

    assert (np.abs(blown_up_mat - final_mat)).sum()/ size_1*size_2 == 0.0

#------------with padding ----------------------


def test_method_76():
    '''
    mat size: 50 X 50
    padding: 10
    patch_size: 30
    patch_tensor: (1,5,5,30,30)

    blown_up_scale = 4

    return : tensor (30,30)

    '''

    blown_up_mat = blown_up_function(matrix= mat0)

    blown_up_patches = blown_up_function(matrix= test_method_4())

    size_1, size_2 , dim_1 , dim_2 = blown_up_patches.shape

    blown_up_patches = blown_up_patches.reshape(1,size_1,size_2, dim_1,dim_2)

    final_mat = unpatching(blown_up_patches = blown_up_patches,
    dimensions = (200,200),
    blown_up_scale = 4)

    assert (np.abs(blown_up_mat - final_mat)).sum()/ size_1*size_2 == 0.0




def test_method_77():
    '''
    mat size: 100 X 100
    padding: 5
    patch_size: 30
    patch_tensor: (1,10,10,30,30)

    blown_up_scale = 4

    return : tensor (30,30)

    '''

    blown_up_mat = blown_up_function(matrix= mat1)

    blown_up_patches = blown_up_function(matrix= test_method_5())

    size_1, size_2 , dim_1 , dim_2 = blown_up_patches.shape

    blown_up_patches = blown_up_patches.reshape(1,size_1,size_2, dim_1,dim_2)

    final_mat = unpatching(blown_up_patches = blown_up_patches,
    dimensions = (400,400),
    blown_up_scale = 4)

    assert (np.abs(blown_up_mat - final_mat)).sum()/ size_1*size_2 == 0.0


def test_method_78():
    '''
    mat size: 250 X 250
    padding: 10
    patch_size: 30
    patch_tensor: (1,25,25,30,30)

    blown_up_scale = 4

    return : tensor (30,30)

    '''

    blown_up_mat = blown_up_function(matrix= mat2)

    blown_up_patches = blown_up_function(matrix= test_method_6())

    size_1, size_2 , dim_1 , dim_2 = blown_up_patches.shape

    blown_up_patches = blown_up_patches.reshape(1,size_1,size_2, dim_1,dim_2)

    final_mat = unpatching(blown_up_patches = blown_up_patches,
    dimensions = (1000,1000),
    blown_up_scale = 4)

    assert (np.abs(blown_up_mat - final_mat)).sum()/ size_1*size_2 == 0.0



def test_method_79():
    '''
    mat size: 500 X 500
    padding: 5
    patch_size: 30
    patch_tensor: (1,50,50,30,30)

    blown_up_scale = 4

    return : tensor (30,30)

    '''

    blown_up_mat = blown_up_function(matrix= mat3)

    blown_up_patches = blown_up_function(matrix= test_method_7())

    size_1, size_2 , dim_1 , dim_2 = blown_up_patches.shape

    blown_up_patches = blown_up_patches.reshape(1,size_1,size_2, dim_1,dim_2)

    final_mat = unpatching(blown_up_patches = blown_up_patches,
    dimensions = (2000,2000),
    blown_up_scale = 4)

    assert (np.abs(blown_up_mat - final_mat)).sum()/ size_1*size_2 == 0.0

#--------------testing with unbalanced tensor----------------

#------------checking with zero padding -----------------------


def test_method_80():
    '''
    mat size: 50 X 100
    padding: 0
    patch_size: 10
    patch_tensor: (1,5,5,10,10)

    blown_up_scale = 4

    return : tensor (10,10)

    '''

    blown_up_mat = blown_up_function(matrix= mat0_un)

    blown_up_patches = blown_up_function(matrix= test_method_8())

    size_1, size_2 , dim_1 , dim_2 = blown_up_patches.shape

    blown_up_patches = blown_up_patches.reshape(1,size_1,size_2, dim_1,dim_2)

    final_mat = unpatching(blown_up_patches = blown_up_patches,
    dimensions = (200,400),
    blown_up_scale = 4)

    assert (np.abs(blown_up_mat - final_mat)).sum()/size_1*size_2 == 0.0


def test_method_81():
    '''
    mat size: 100 X 200
    padding: 0
    patch_size: 10
    patch_tensor: (1,10,20,10,10)

    blown_up_scale = 4

    return : tensor (10,10)

    '''

    blown_up_mat = blown_up_function(matrix = mat1_un)

    blown_up_patches = blown_up_function(matrix = test_method_9())

    size_1, size_2 , dim_1 , dim_2 = blown_up_patches.shape

    blown_up_patches = blown_up_patches.reshape(1,size_1,size_2, dim_1,dim_2)

    final_mat = unpatching(blown_up_patches = blown_up_patches,
    dimensions = (400,800),
    blown_up_scale = 4)

    assert (np.abs(blown_up_mat - final_mat)).sum()/size_1*size_2 == 0.0


def test_method_82():
    '''
    mat size: 250 X 500
    padding: 0
    patch_size: 10
    patch_tensor: (1,25,50,10,10)

    blown_up_scale = 4

    return : tensor (10,10)

    '''

    blown_up_mat = blown_up_function(matrix = mat2_un)

    blown_up_patches = blown_up_function(matrix = test_method_10())

    size_1, size_2 , dim_1 , dim_2 = blown_up_patches.shape

    blown_up_patches = blown_up_patches.reshape(1,size_1,size_2, dim_1,dim_2)

    final_mat = unpatching(blown_up_patches = blown_up_patches,
    dimensions = (1000,2000),
    blown_up_scale = 4)

    assert (np.abs(blown_up_mat - final_mat)).sum()/size_1*size_2 == 0.0


def test_method_83():
    '''
    mat size: 500 X 1000
    padding: 0
    patch_size: 10
    patch_tensor: (1,50,100,10,10)

    blown_up_scale = 4

    return : tensor (10,10)

    '''

    blown_up_mat = blown_up_function(matrix = mat3_un)

    blown_up_patches = blown_up_function(matrix = test_method_11())

    size_1, size_2 , dim_1 , dim_2 = blown_up_patches.shape

    blown_up_patches = blown_up_patches.reshape(1,size_1,size_2, dim_1,dim_2)

    final_mat = unpatching(blown_up_patches = blown_up_patches,
    dimensions = (2000,4000),
    blown_up_scale = 4)

    assert (np.abs(blown_up_mat - final_mat)).sum()/size_1*size_2 == 0.0

#------------checking with padding -----------------------

def test_method_84():
    '''
    mat size: 50 X 100
    padding: 10
    patch_size: 30
    patch_tensor: (1,5,5,30,30)

    blown_up_scale = 4

    return : tensor (10,10)

    '''

    blown_up_mat = blown_up_function(matrix= mat0_un)

    blown_up_patches = blown_up_function(matrix= test_method_12())

    size_1, size_2 , dim_1 , dim_2 = blown_up_patches.shape

    blown_up_patches = blown_up_patches.reshape(1,size_1,size_2, dim_1,dim_2)

    final_mat = unpatching(blown_up_patches = blown_up_patches,
    dimensions = (200,400),
    blown_up_scale = 4)

    assert (np.abs(blown_up_mat - final_mat)).sum()/size_1*size_2 == 0.0


def test_method_85():
    '''
    mat size: 100 X 200
    padding: 5
    patch_size: 30
    patch_tensor: (1,10,20,30,30)

    blown_up_scale = 4

    return : tensor (10,10)

    '''

    blown_up_mat = blown_up_function(matrix= mat1_un)

    blown_up_patches = blown_up_function(matrix= test_method_13())

    size_1, size_2 , dim_1 , dim_2 = blown_up_patches.shape

    blown_up_patches = blown_up_patches.reshape(1,size_1,size_2, dim_1,dim_2)

    final_mat = unpatching(blown_up_patches = blown_up_patches,
    dimensions = (400,800),
    blown_up_scale = 4)

    assert (np.abs(blown_up_mat - final_mat)).sum()/size_1*size_2 == 0.0


def test_method_86():
    '''
    mat size: 250 X 500
    padding: 10
    patch_size: 30
    patch_tensor: (1,25,50,30,30)

    blown_up_scale = 4

    return : tensor (10,10)

    '''

    blown_up_mat = blown_up_function(matrix= mat2_un)

    blown_up_patches = blown_up_function(matrix= test_method_14())

    size_1, size_2 , dim_1 , dim_2 = blown_up_patches.shape

    blown_up_patches = blown_up_patches.reshape(1,size_1,size_2, dim_1,dim_2)

    final_mat = unpatching(blown_up_patches = blown_up_patches,
    dimensions = (1000,2000),
    blown_up_scale = 4)

    assert (np.abs(blown_up_mat - final_mat)).sum()/size_1*size_2 == 0.0


def test_method_87():
    '''
    mat size: 500 X 1000
    padding: 5
    patch_size: 30
    patch_tensor: (1,25,50,30,30)

    blown_up_scale = 4

    return : tensor (10,10)

    '''

    blown_up_mat = blown_up_function(matrix= mat3_un)

    blown_up_patches = blown_up_function(matrix= test_method_15())

    size_1, size_2 , dim_1 , dim_2 = blown_up_patches.shape

    blown_up_patches = blown_up_patches.reshape(1,size_1,size_2, dim_1,dim_2)

    final_mat = unpatching(blown_up_patches = blown_up_patches,
    dimensions = (2000,4000),
    blown_up_scale = 4)

    assert (np.abs(blown_up_mat - final_mat)).sum()/size_1*size_2 == 0.0



#testing for three channel pathces -----------------------

def blown_up_function_3d_pathces(matrix):
    '''
    The function takes the patches or 3 dim matrix, and returns the blown up patches or matrix

    Arguments:
    patches - either single channel or three channel, dim (channel , length, width, length of patch, width of patch)

    ------------------------------------------------------
    return :

    blown up patches - the patches are enlarged by 4 times , using nearest algo
    '''

    extra, shape_1, shape_2 , dim_1, dim_2 = matrix.shape

    blown_1 = matrix[0]
    blown_2= matrix[1]
    blown_3 = matrix[2]

    matrix_1 = torch.tensor(blown_1)
    matrix_2 = torch.tensor(blown_2)
    matrix_3 = torch.tensor(blown_3)
    scale_factor = 4
    scaling = nn.Upsample(scale_factor = scale_factor, mode = 'nearest')

    blown_up_1 = scaling(matrix_1)
    blown_up_2 = scaling(matrix_2)
    blown_up_3 = scaling(matrix_3)

    blown_up_1 = blown_up_1.reshape(1,blown_up_1.shape[0],
    blown_up_1.shape[1],
    blown_up_1.shape[2],
    blown_up_1.shape[3])
    blown_up_2 = blown_up_2.reshape(1,blown_up_2.shape[0],
    blown_up_2.shape[1],
    blown_up_2.shape[2],
    blown_up_2.shape[3])
    blown_up_3 = blown_up_3.reshape(1,blown_up_3.shape[0],
    blown_up_3.shape[1],
    blown_up_3.shape[2],
    blown_up_3.shape[3])

    combined_patch = torch.cat( (blown_up_1, blown_up_2,blown_up_3), 0  )

    return combined_patch.numpy()

#blowing up the 3d matrix

def blow_up_3d_matrix(matrix):

    '''
    the function takes the 3 channel matrix and enlarges it 4 times and returns the blown up matrix

    Arguments:

        matrix - a three dimensionsal matrix (channel, dim_1 , dim_2)
    -------------------------------------
    retrun:

        matrix - blown up by 4 times
    '''

    size_1, dim_1, dim_2 = matrix.shape
    matrix = torch.tensor(matrix)

    matrix = matrix.view(1, size_1, dim_1,dim_2)
    scaling = nn.Upsample(scale_factor = 4, mode = 'nearest')
    matrix = scaling(matrix)
    matrix = matrix.numpy()

    return matrix



#testing with zero padding ---------------------------

def test_method_88():
    '''
    mat size: 3 X 50 X 50
    padding: 0
    patch_size: 10
    patch_tensor: (1,5,5,10,10)

    blown_up_scale = 4

    return : abs diffrence (initial - combined)

    '''
    mat = blow_up_3d_matrix(matrix = mat0_3d) #change

    blown_patches = blown_up_function_3d_pathces(matrix = test_method_16()) #change

    final_mat = unpatching(blown_up_patches = blown_patches,
    dimensions = (200,200),  #change
    blown_up_scale = 4)

    extra, dim_1, dim_2 = final_mat.shape


    assert (np.abs(mat - final_mat)).sum()/dim_1*dim_2 == 0.0



def test_method_89():
    '''
    mat size: 3 X 100 X 100
    padding: 0
    patch_size: 10
    patch_tensor: (1,10,10,10,10)

    blown_up_scale = 4

    return : abs diffrence (initial - combined)

    '''
    mat = blow_up_3d_matrix(matrix = mat1_3d) #change

    blown_patches = blown_up_function_3d_pathces(matrix = test_method_17()) #change

    final_mat = unpatching(blown_up_patches = blown_patches,
    dimensions = (400,400),  #change
    blown_up_scale = 4)

    extra, dim_1, dim_2 = final_mat.shape


    assert (np.abs(mat - final_mat)).sum()/dim_1*dim_2 == 0.0


def test_method_90():
    '''
    mat size: 3 X 250 X 250
    padding: 0
    patch_size: 10
    patch_tensor: (1,25,25,10,10)

    blown_up_scale = 4

    return : abs diffrence (initial - combined)

    '''
    mat = blow_up_3d_matrix(matrix = mat2_3d) #change

    blown_patches = blown_up_function_3d_pathces(matrix = test_method_18()) #change

    final_mat = unpatching(blown_up_patches = blown_patches,
    dimensions = (1000,1000),  #change
    blown_up_scale = 4)

    extra, dim_1, dim_2 = final_mat.shape


    assert (np.abs(mat - final_mat)).sum()/dim_1*dim_2 == 0.0


def test_method_91():
    '''
    mat size: 3 X 500 X 500
    padding: 0
    patch_size: 10
    patch_tensor: (1,50,50,10,10)

    blown_up_scale = 4

    return : abs diffrence (initial - combined)

    '''
    mat = blow_up_3d_matrix(matrix = mat3_3d) #change

    blown_patches = blown_up_function_3d_pathces(matrix = test_method_19()) #change

    final_mat = unpatching(blown_up_patches = blown_patches,
    dimensions = (2000,2000),  #change
    blown_up_scale = 4)

    extra, dim_1, dim_2 = final_mat.shape


    assert (np.abs(mat - final_mat)).sum()/dim_1*dim_2 == 0.0


#-------------checking with padding ---------------------------------------------


def test_method_92():
    '''
    mat size: 3 X 50 X 50
    padding: 10
    patch_size: 30
    patch_tensor: (1,5,5,30,30)

    blown_up_scale = 4

    return : abs diffrence (initial - combined)

    '''
    mat = blow_up_3d_matrix(matrix = mat0_3d) #change

    blown_patches = blown_up_function_3d_pathces(matrix = test_method_20()) #change

    final_mat = unpatching(blown_up_patches = blown_patches,
    dimensions = (200,200),  #change
    blown_up_scale = 4)

    extra, dim_1, dim_2 = final_mat.shape


    assert (np.abs(mat - final_mat)).sum()/dim_1*dim_2 == 0.0


def test_method_93():
    '''
    mat size: 3 X 100 X 100
    padding: 5
    patch_size: 30
    patch_tensor: (1,10,10,30,30)

    blown_up_scale = 4

    return : abs diffrence (initial - combined)

    '''
    mat = blow_up_3d_matrix(matrix = mat1_3d) #change

    blown_patches = blown_up_function_3d_pathces(matrix = test_method_21()) #change

    final_mat = unpatching(blown_up_patches = blown_patches,
    dimensions = (400,400),  #change
    blown_up_scale = 4)

    extra, dim_1, dim_2 = final_mat.shape


    assert (np.abs(mat - final_mat)).sum()/dim_1*dim_2 == 0.0



def test_method_94():
    '''
    mat size: 3 X 250 X 250
    padding: 10
    patch_size: 30
    patch_tensor: (1,25,25,30,30)

    blown_up_scale = 4

    return : abs diffrence (initial - combined)

    '''
    mat = blow_up_3d_matrix(matrix = mat2_3d) #change

    blown_patches = blown_up_function_3d_pathces(matrix = test_method_22()) #change

    final_mat = unpatching(blown_up_patches = blown_patches,
    dimensions = (1000,1000),  #change
    blown_up_scale = 4)

    extra, dim_1, dim_2 = final_mat.shape


    assert (np.abs(mat - final_mat)).sum()/dim_1*dim_2 == 0.0


def test_method_95():
    '''
    mat size: 3 X 500 X 500
    padding: 5
    patch_size: 30
    patch_tensor: (1,25,25,30,30)

    blown_up_scale = 4

    return : abs diffrence (initial - combined)

    '''
    mat = blow_up_3d_matrix(matrix = mat3_3d) #change

    blown_patches = blown_up_function_3d_pathces(matrix = test_method_23()) #change

    final_mat = unpatching(blown_up_patches = blown_patches,
    dimensions = (2000,2000),  #change
    blown_up_scale = 4)

    extra, dim_1, dim_2 = final_mat.shape

    assert (np.abs(mat - final_mat)).sum()/dim_1*dim_2 == 0.0



#checking with unbalanced tensor with zero padding ------------------------

def test_method_96():
    '''
    mat size: 3 X 50 X 100
    padding: 0
    patch_size: 10
    patch_tensor: (1,5,10,10,10)

    blown_up_scale = 4

    return : abs diffrence (initial - combined)

    '''
    mat = blow_up_3d_matrix(matrix = mat0_3d_un) #change

    blown_patches = blown_up_function_3d_pathces(matrix = test_method_24()) #change

    final_mat = unpatching(blown_up_patches = blown_patches,
    dimensions = (200,400),  #change
    blown_up_scale = 4)

    extra, dim_1, dim_2 = final_mat.shape

    assert (np.abs(mat - final_mat)).sum()/dim_1*dim_2 == 0.0


def test_method_97():
    '''
    mat size: 3 X 100 X 200
    padding: 0
    patch_size: 10
    patch_tensor: (1,10,20,10,10)

    blown_up_scale = 4

    return : abs diffrence (initial - combined)

    '''
    mat = blow_up_3d_matrix(matrix = mat1_3d_un) #change

    blown_patches = blown_up_function_3d_pathces(matrix = test_method_25()) #change

    final_mat = unpatching(blown_up_patches = blown_patches,
    dimensions = (400,800),  #change
    blown_up_scale = 4)

    extra, dim_1, dim_2 = final_mat.shape

    assert (np.abs(mat - final_mat)).sum()/dim_1*dim_2 == 0.0


def test_method_98():
    '''
    mat size: 3 X 250 X 500
    padding: 0
    patch_size: 10
    patch_tensor: (1,25,50,10,10)

    blown_up_scale = 4

    return : abs diffrence (initial - combined)

    '''
    mat = blow_up_3d_matrix(matrix = mat2_3d_un) #change

    blown_patches = blown_up_function_3d_pathces(matrix = test_method_26()) #change

    final_mat = unpatching(blown_up_patches = blown_patches,
    dimensions = (1000,2000),  #change
    blown_up_scale = 4)

    extra, dim_1, dim_2 = final_mat.shape

    assert (np.abs(mat - final_mat)).sum()/dim_1*dim_2 == 0.0


def test_method_99():
    '''
    mat size: 3 X 500 X 1000
    padding: 0
    patch_size: 10
    patch_tensor: (1,50,100,10,10)

    blown_up_scale = 4

    return : abs diffrence (initial - combined)

    '''
    mat = blow_up_3d_matrix(matrix = mat3_3d_un) #change

    blown_patches = blown_up_function_3d_pathces(matrix = test_method_27()) #change

    final_mat = unpatching(blown_up_patches = blown_patches,
    dimensions = (2000,4000),  #change
    blown_up_scale = 4)

    extra, dim_1, dim_2 = final_mat.shape

    assert (np.abs(mat - final_mat)).sum()/dim_1*dim_2 == 0.0

#testing with overlapping patches -------------------------------------


def test_method_100():
    '''
    mat size: 3 X 50 X 100
    padding: 10
    patch_size: 30
    patch_tensor: (1,5,10,30,30)

    blown_up_scale = 4

    return : abs diffrence (initial - combined)

    '''
    mat = blow_up_3d_matrix(matrix = mat0_3d_un) #change

    blown_patches = blown_up_function_3d_pathces(matrix = test_method_28()) #change

    final_mat = unpatching(blown_up_patches = blown_patches,
    dimensions = (200,400),  #change
    blown_up_scale = 4)

    extra, dim_1, dim_2 = final_mat.shape

    assert (np.abs(mat - final_mat)).sum()/dim_1*dim_2 == 0.0


def test_method_101():
    '''
    mat size: 3 X 100 X 200
    padding: 5
    patch_size: 30
    patch_tensor: (1,10,20,30,30)

    blown_up_scale = 4

    return : abs diffrence (initial - combined)

    '''
    mat = blow_up_3d_matrix(matrix = mat1_3d_un) #change

    blown_patches = blown_up_function_3d_pathces(matrix = test_method_29()) #change

    final_mat = unpatching(blown_up_patches = blown_patches,
    dimensions = (400,800),  #change
    blown_up_scale = 4)

    extra, dim_1, dim_2 = final_mat.shape

    assert (np.abs(mat - final_mat)).sum()/dim_1*dim_2 == 0.0


def test_method_102():
    '''
    mat size: 3 X 250 X 500
    padding: 10
    patch_size: 30
    patch_tensor: (1,25,50,30,30)

    blown_up_scale = 4

    return : abs diffrence (initial - combined)

    '''
    mat = blow_up_3d_matrix(matrix = mat2_3d_un) #change

    blown_patches = blown_up_function_3d_pathces(matrix = test_method_30()) #change

    final_mat = unpatching(blown_up_patches = blown_patches,
    dimensions = (1000,2000),  #change
    blown_up_scale = 4)

    extra, dim_1, dim_2 = final_mat.shape

    assert (np.abs(mat - final_mat)).sum()/dim_1*dim_2 == 0.0


def test_method_103():
    '''
    mat size: 3 X 500 X 1000
    padding: 5
    patch_size: 30
    patch_tensor: (1,50,100,30,30)

    blown_up_scale = 4

    return : abs diffrence (initial - combined)

    '''
    mat = blow_up_3d_matrix(matrix = mat3_3d_un) #change

    blown_patches = blown_up_function_3d_pathces(matrix = test_method_31()) #change

    final_mat = unpatching(blown_up_patches = blown_patches,
    dimensions = (2000,4000),  #change
    blown_up_scale = 4)

    extra, dim_1, dim_2 = final_mat.shape

    assert (np.abs(mat - final_mat)).sum()/dim_1*dim_2 == 0.0

#checking error cases -----------------------------------


def test_method_104():
    '''
    mat size: 500 X 500
    padding: 0
    patch_size: 10

    return : error
    '''

    patch_3 = patching_input(start_tensor=  mat3,
    patch_size = 0)

    assert patch_3 == "Enter a non zero patch size"

