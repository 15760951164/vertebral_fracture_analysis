import sys
import argparse
from glob import glob
import os
import numpy as np
import itk
import multiprocessing


def reorient_to_rai(image):
    """
    Reorient image to RAI orientation.
    :param image: Input itk image.
    :return: Input image reoriented to RAI.
    """
    filter = itk.OrientImageFilter.New(image)
    filter.UseImageDirectionOn()
    filter.SetInput(image)
    m = itk.GetMatrixFromArray(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64))
    filter.SetDesiredCoordinateDirection(m)
    filter.Update()
    reoriented = filter.GetOutput()
    return reoriented


def smooth(image, sigma):
    """
    Smooth image with Gaussian smoothing.
    :param image: ITK image.
    :param sigma: Sigma for smoothing.
    :return: Smoothed image.
    """
    ImageType = itk.Image[itk.SS, 3]
    filter = itk.SmoothingRecursiveGaussianImageFilter[ImageType, ImageType].New()
    filter.SetInput(image)
    filter.SetSigma(sigma)
    filter.Update()
    smoothed = filter.GetOutput()
    return smoothed

def clamp(image):
    """
    Clamp image between -1024 to 8192.
    :param image: ITK image.
    :return: Clamped image.
    """
    ImageType = itk.Image[itk.SS, 3]
    filter = itk.ClampImageFilter[ImageType, ImageType].New()
    filter.SetInput(image)
    filter.SetBounds(-2048, 8192)
    filter.Update()
    clamped = filter.GetOutput()
    return clamped

def process_image(filename, output_folder, sigma):
    """
    Reorient image at filename, smooth with sigma, clamp and save to output_folder.
    :param filename: The image filename.
    :param output_folder: The output folder.
    :param sigma: Sigma for smoothing.
    """
    basename = os.path.basename(filename)
    basename_wo_ext = basename[:basename.find('.nii.gz')]
    print(basename_wo_ext)
    ImageType = itk.Image[itk.SS, 3]
    reader = itk.ImageFileReader[ImageType].New()
    reader.SetFileName(filename)
    image = reader.GetOutput()
    reoriented = reorient_to_rai(image)
    if not basename_wo_ext.endswith('_seg'):
        reoriented = smooth(reoriented, sigma)
        #reoriented = clamp(reoriented)
    reoriented.SetOrigin([0, 0, 0])
    m = itk.GetMatrixFromArray(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64))
    reoriented.SetDirection(m)
    reoriented.Update()
    itk.imwrite(reoriented, os.path.join(output_folder, basename_wo_ext + '.nii.gz'))

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, default="data/images")
    parser.add_argument('--output_folder', type=str, default="data/images_reoriented")
    parser.add_argument('--sigma', type=float, default=0.75)
    parser_args = parser.parse_args()
    
    check_dir(parser_args.output_folder)
    
    for id in os.listdir(parser_args.image_folder):
        filenames = glob(os.path.join(parser_args.image_folder, id, '*.nii.gz'))
        for filename in sorted(filenames):
            output_path = os.path.join(parser_args.output_folder, id)
            check_dir(output_path)
            process_image(filename, output_path, parser_args.sigma)
    
    # pool = multiprocessing.Pool(8)
    # pool.starmap(process_image, [(filename, parser_args.output_folder, parser_args.sigma) for filename in sorted(filenames)])