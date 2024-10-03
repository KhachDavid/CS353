import numpy as np
import pickle
import f

def test_loadCameraman(expected_outputs):
    result = np.zeros(5, dtype=bool)
    ############################### TEST CASE 0 ###############################
    # Test case where the input size is (128, 128)
    imsize = (128, 128)
    # Load the expected output
    expected_output = expected_outputs['test_loadCameraman'][0]
    # Call the function
    output = f.loadCameraman(imsize)
    # Test the output
    result[0] = np.allclose(output, expected_output)
    ############################### TEST CASE 1 ###############################
    # Test case where the input size is (512, 512)
    imsize = (512, 512)
    # Load the expected output
    expected_output = expected_outputs['test_loadCameraman'][1]
    # Call the function
    output = f.loadCameraman(imsize)
    # Test the output
    result[1] = np.allclose(output, expected_output)
    ############################### TEST CASE 2 ###############################
    # Test case where the input size is (256, 256)
    imsize = (256, 256)
    # Load the expected output
    expected_output = expected_outputs['test_loadCameraman'][2]
    # Call the function
    output = f.loadCameraman(imsize)
    # Test the output
    result[2] = np.allclose(output, expected_output)
    ############################### TEST CASE 3 ###############################
    # Test case where the input size is (64, 128)
    imsize = (64, 128)
    # Load the expected output
    expected_output = expected_outputs['test_loadCameraman'][3]
    # Call the function
    output = f.loadCameraman(imsize)
    # Test the output
    result[3] = np.allclose(output, expected_output)
    ############################### TEST CASE 4 ###############################
    # Test case where the input size is (128, 64)
    imsize = (128, 64)
    # Load the expected output
    expected_output = expected_outputs['test_loadCameraman'][4]
    # Call the function
    output = f.loadCameraman(imsize)
    # Test the output
    result[4] = np.allclose(output, expected_output)
    ############################### END OF TEST ###############################
    return result

def test_generate2DGaussianKernel(expected_outputs):
    result = np.zeros(5, dtype=bool)
    ############################### TEST CASE 0 ###############################
    # Test case where the kernel size is (3, 3) and standard deviation is 1
    kernel_size = (3, 3)
    std = 1
    # Load the expected output
    expected_output = expected_outputs['test_generate2DGaussianKernel'][0]
    # Call the function
    output = f.generate2DGaussianKernel(kernel_size, std)
    # Test the output
    result[0] = np.allclose(output, expected_output)
    ############################### TEST CASE 1 ###############################
    # Test case where the kernel size is (5, 5) and standard deviation is 3
    kernel_size = (15, 15)
    std = 3
    # Load the expected output
    expected_output = expected_outputs['test_generate2DGaussianKernel'][1]
    # Call the function
    output = f.generate2DGaussianKernel(kernel_size, std)
    # Test the output
    result[1] = np.allclose(output, expected_output)
    ############################### TEST CASE 2 ###############################
    # Test case where the kernel size is (21, 21) and standard deviation is 3
    kernel_size = (21, 21)
    std = 3
    # Load the expected output
    expected_output = expected_outputs['test_generate2DGaussianKernel'][2]
    # Call the function
    output = f.generate2DGaussianKernel(kernel_size, std)
    # Test the output
    result[2] = np.allclose(output, expected_output)
    ############################### TEST CASE 3 ###############################
    # Test case where the kernel size is (3, 5) and standard deviation is 1
    kernel_size = (3, 5)
    std = 1
    # Load the expected output
    expected_output = expected_outputs['test_generate2DGaussianKernel'][3]
    # Call the function
    output = f.generate2DGaussianKernel(kernel_size, std)
    # Test the output
    result[3] = np.allclose(output, expected_output)
    ############################### TEST CASE 4 ###############################
    # Test case where the kernel size is (5, 3) and standard deviation is 1
    kernel_size = (5, 3)
    std = 1
    # Load the expected output
    expected_output = expected_outputs['test_generate2DGaussianKernel'][4]
    # Call the function
    output = f.generate2DGaussianKernel(kernel_size, std)
    # Test the output
    result[4] = np.allclose(output, expected_output)
    ############################### END OF TEST ###############################
    return result

def test_spatial2DConvolution(expected_outputs):
    result = np.zeros(3, dtype=bool)
    ############################### TEST CASE 0 ###############################
    # Test case where the image size is (3, 3) and kernel size is (3, 3)
    image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    # Load the expected output
    expected_output = expected_outputs['test_spatial2DConvolution'][0]
    # Call the function
    output = f.spatial2DConvolution(image, kernel)
    # Test the output
    result[0] = np.allclose(output, expected_output)
    ############################### TEST CASE 1 ###############################
    # Test case where the image size is (5, 5) and kernel size is (3, 3)
    image = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]])
    kernel = np.array([[10, 20, 10], [20, 10, 20], [100, 300, 500]])
    # Load the expected output
    expected_output = expected_outputs['test_spatial2DConvolution'][1]
    # Call the function
    output = f.spatial2DConvolution(image, kernel)
    # Test the output
    result[1] = np.allclose(output, expected_output)
    ############################### TEST CASE 2 ###############################
    # Test case where the image size is (11, 21) and kernel size is (3, 7)
    image = np.arange(231).reshape(11, 21)
    kernel = np.arange(21).reshape(3, 7)
    # Load the expected output
    expected_output = expected_outputs['test_spatial2DConvolution'][2]
    # Call the function
    output = f.spatial2DConvolution(image, kernel)
    # Test the output
    result[2] = np.allclose(output, expected_output)
    ############################### END OF TEST ###############################
    return result

def test_brightnessHist(expected_outputs):
    result = np.zeros(2, dtype=bool)
    ############################### TEST CASE 0 ###############################
    # Test case where the input image is cameraman image
    image = expected_outputs['test_loadCameraman'][1]
    # Load the expected output
    expected_output = expected_outputs['test_brightnessHist'][0]
    # Call the function
    output = f.brightnessHist(image, (0,1), 101, True)
    # Test the output
    tc0a = np.allclose(output[0], expected_output[0])
    tc0b = np.allclose(output[1], expected_output[1])
    result[0] = np.logical_and(tc0a, tc0b)
    ############################### TEST CASE 1 ###############################
    # Test case where the input image is cameraman image
    q = np.arange(120).reshape(10, 12)
    r = np.arange(156).reshape(12, 13)
    image = q @ r
    # Load the expected output
    expected_output = expected_outputs['test_brightnessHist'][1]
    # Call the function
    output = f.brightnessHist(image, (np.min(image), np.max(image)), 10, True)
    # Test the output
    tc1a = np.allclose(output[0], expected_output[0])
    tc1b = np.allclose(output[1], expected_output[1])
    result[1] = np.logical_and(tc1a, tc1b)
    ############################### END OF TEST ###############################
    return result

def autograde():
    # Load the expected outputs
    with open('autograder_results.pkl', 'rb') as file:
        expected_outputs = pickle.load(file)
    # Run the test functions
    result_loadCameraman = test_loadCameraman(expected_outputs)
    result_generate2DGaussianKernel = test_generate2DGaussianKernel(expected_outputs)
    result_spatial2DConvolution = test_spatial2DConvolution(expected_outputs)
    result_brightnessHist = test_brightnessHist(expected_outputs)
    grade = 0
    if np.all(result_loadCameraman):
        section_grade = 10
        print('loadCameraman... \t\tPASS \t', section_grade,'/10')
        grade += section_grade
    else:
        section_grade = np.count_nonzero(result_loadCameraman) * 2
        print('loadCameraman... \t\tFAIL\t', section_grade,'/10')
        print(result_loadCameraman)
        grade += section_grade
    if np.all(result_generate2DGaussianKernel):
        section_grade = 15
        print('generate2DGaussianKernel... \tPASS \t', section_grade,'/15')
        grade += section_grade
    else:
        section_grade = np.count_nonzero(result_generate2DGaussianKernel) * 3
        print('generate2DGaussianKernel... \tFAIL\t', section_grade,'/15')
        print(result_generate2DGaussianKernel)
        grade += section_grade
    if np.all(result_spatial2DConvolution):
        section_grade = 5
        print('spatial2DConvolution... \tPASS \t', section_grade,'/5')
        grade += section_grade
    else:
        section_grade = np.count_nonzero(result_spatial2DConvolution) * 1.67
        print('spatial2DConvolution... \tFAIL\t', section_grade,'/5')
        print(result_spatial2DConvolution)
        grade += section_grade
    if np.all(result_brightnessHist):
        section_grade = 10
        print('brightnessHist... \t\tPASS \t', section_grade,'/10')
        grade += section_grade
    else:
        section_grade = np.count_nonzero(result_brightnessHist) * 5
        print('brightnessHist... \tFAIL\t', section_grade,'/10')
        print(result_brightnessHist)
        grade += section_grade
    print('Expected Grade:', grade, '/ 40')
    
    print('')
    print('Note that this grade is intended to serve as a guide and is not final.')
    print('The final grade will be assigned using a different autograder.')
    print('Your plots and open ended questions will be reviewed manually.')
