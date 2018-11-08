#-*-coding:utf-8-*-
import importlib
import sys
importlib.reload(sys)
import cv2
import numpy
import pydicom
from matplotlib import pyplot as plt
import os
import glob
import numpy as np
import SimpleITK as sitk
import skimage
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    image[image == -2000] = 0
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)
    return np.array(image, dtype=np.int16)

def convert_dcm_to_mhd(dcmFilesList, DirName, SavePath):
    RefDs = pydicom.read_file(dcmFilesList[0])  
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(dcmFilesList))
    ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))
    Origin = RefDs.ImagePositionPatient
    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
    ArrayDicom = load_scan(DirName)
    ArrayDicom = get_pixels_hu(ArrayDicom)
    sitk_img = sitk.GetImageFromArray(ArrayDicom, isVector=False)
    sitk_img.SetSpacing(ConstPixelSpacing)
    sitk_img.SetOrigin(Origin)
    sitk.WriteImage(sitk_img, SavePath)

def get_segmented_lungs(im, plot=False):
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    if plot == True:
        f, plots = plt.subplots(8, 1, figsize=(5, 40))
    '''
    Step 1: Convert into a binary image.
    '''
    binary = im < -320
    if plot == True:
        plots[0].axis('off')
        plots[0].set_title('binary image')
        plots[0].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].set_title('after clear border')
        plots[1].imshow(cleared, cmap=plt.cm.bone)
    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].set_title('found all connective graph')
        plots[2].imshow(label_image, cmap=plt.cm.bone)
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[3].axis('off')
        plots[3].set_title(' Keep the labels with 2 largest areas')
        plots[3].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].set_title('seperate the lung nodules attached to the blood vessels')
        plots[4].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].set_title('keep nodules attached to the lung wall')
        plots[5].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].set_title('Fill in the small holes inside the binary mask of lungs')
        plots[6].imshow(binary, cmap=plt.cm.bone)
    return binary

from scipy import ndimage as ndi
import scipy.misc
def resample_mhd(image, space, new_spacing=[1,1,1]):
    spacing = map(float, ([space[2]] + [space[0]] + [space[1]]))
    spacing = np.array(list(spacing))
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    return image, new_spacing

def generate_3DLung_ply(ply_path, file_path, threshold=0, rgb=[189, 223, 235]):
    if not os.path.exists(ply_path):
        os.mkdir(ply_path)
    img = sitk.ReadImage(file_path)
    Spacing = img.GetSpacing()
    spacing = np.array(Spacing)
    origin = np.array(img.GetOrigin())
    img_array = sitk.GetArrayFromImage(img)
    img_array_new, new_space = resample_mhd(img_array, Spacing, new_spacing=[1, 1, 1])
    img_shape = np.array(img_array.transpose(2, 1, 0).shape)
    p = img_array_new.transpose(2, 1, 0)
    verts, faces = measure.marching_cubes_classic(p, threshold)
    file_name = file_path[-22: -4]
    writeply(ply_path + file_name + '.ply',verts, faces, rgb)

def writeply(filename, points, pieces, rgb=[152, 152, 152]):
    target = open(filename, 'w')
    target.write('ply\n');
    target.write('format ascii 1.0 \n')
    target.write('element vertex ' + str(points.shape[0]) + '\n')
    target.write('property float x\n')
    target.write('property float y\n')
    target.write('property float z\n')
    target.write('property uchar red\n')
    target.write('property uchar green\n')
    target.write('property uchar blue\n')
    target.write('end_header\n')
    for i in range(points.shape[0]):
        target.write('%f %f %f %d %d %d\n'%(points[i,0],points[i,1],points[i,2], rgb[0],rgb[1],rgb[2]))
    for j in range(pieces.shape[0]):
        target.write('3 %d %d %d\n' % (pieces[j, 0], pieces[j, 1], pieces[j, 2]))
    target.close()


if __name__ == "__main__":
    mask_file_path = r'E:\LeStudy\MyPythonCode\PyCharmProjects\src\Lung\DICOM_Seg\test\lung_mask_SaveRaw'
    patient_name = glob.glob(mask_file_path+'/'+'*.mhd')
    LungMask_ply_Path = './LungMask_ply/'
    if not os.path.exists(LungMask_ply_Path):
        os.mkdir(LungMask_ply_Path)
    for patient in patient_name:
        print('Processing: ', patient)
        generate_3DLung_ply(LungMask_ply_Path, patient, 0)
    print('Generate PLY Finished!')