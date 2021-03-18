#%% Initialize
import napari
import numpy as np

from pystackreg import StackReg

from joblib import Parallel, delayed  

from skimage import io
from skimage.util import invert
from skimage.filters import sato
from skimage.measure import regionprops
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.morphology import disk, black_tophat

from scipy.signal import medfilt
from scipy.ndimage.morphology import distance_transform_edt

#%% varnames
ROOTPATH = 'D:/CurrentTasks/CENTURIProject_LAI_ClaireValotteau/'
RAWNAME = 'raw_01_c02.tif'
EMPTY_t0 = 2020

CIRCLES_n = 1
CIRCLES_min_rad = 35
CIRCLES_max_rad = 45

#%% Open Stack from RAWNAME

raw = io.imread(ROOTPATH+RAWNAME)
nT = raw.shape[0] # Get Stack dimension (t)
nY = raw.shape[1] # Get Stack dimension (x)
nX = raw.shape[2] # Get Stack dimension (y)

#%% image processing

def image_process(im):
    '''Enter function general description + arguments'''
    strel = disk(5) 
    im_wavg = np.mean(im,0)
    im_binary = black_tophat(im_wavg, strel)
    im_binary = im_binary > np.max(im_binary)/2
    im_binary = im_binary.astype('float')
    return im_wavg, im_binary

output_list = Parallel(n_jobs=35)(
    delayed(image_process)(
        raw[i:i+30,:,:]
        )
    for i in range(nT-30)
    )
 
raw_wavg = np.stack([arrays[0] for arrays in output_list], axis=0)
raw_wavg_binary = np.stack([arrays[1] for arrays in output_list], axis=0)

#%% image registration (background)

def image_reg(im0, im1, im2reg):
    '''Enter function general description + arguments'''
    sr = StackReg(StackReg.TRANSLATION)
    sr.register(im0, im1)
    im_reg = sr.transform(im2reg)
    return im_reg

output_list = Parallel(n_jobs=35)(
    delayed(image_reg)(
        raw_wavg_binary[0,:,:],
        raw_wavg_binary[i,:,:],
        raw_wavg[i,:,:]) 
    for i in range(nT-30)
    )
 
raw_wavg_reg = np.stack([arrays for arrays in output_list], axis=0)   

#%% subtract static background

static_bg = np.mean(raw_wavg_reg[EMPTY_t0:-1,:,:],0)
static_bg[static_bg == 0] = 'nan'
raw_wavg_reg_bgsub = raw_wavg_reg - static_bg
raw_wavg_reg_bgsub = np.nan_to_num(raw_wavg_reg_bgsub, nan=0.0) 

#%% apply sato filter

def sato_filter(im):
    '''Enter function general description + arguments'''
    im_sato = sato(im,sigmas=4,mode='reflect',black_ridges=False)   
    return im_sato

output_list = Parallel(n_jobs=35)(
    delayed(sato_filter)(
        raw_wavg_reg_bgsub[i,:,:]
        ) 
    for i in range(nT-30)
    ) 

raw_wavg_reg_bgsub_sato = np.stack([arrays for arrays in output_list], axis=0)  

#%% image registration (bead)

thresh_quant = np.quantile(raw_wavg_reg_bgsub_sato, 0.95)
raw_wavg_reg_bgsub_sato = raw_wavg_reg_bgsub_sato > thresh_quant
raw_wavg_reg_bgsub_sato = raw_wavg_reg_bgsub_sato.astype('float') 

props = regionprops(raw_wavg_reg_bgsub_sato[0,:,:].astype('int'))
ctrd_x = props.centroid[1]
ctrd_y = props.centroid[0]
           
def image_reg(im0, im1, im2reg):
    '''Enter function general description + arguments'''
    sr = StackReg(StackReg.TRANSLATION)
    sr.register(im0, im1)
    im_reg = sr.transform(im2reg)
    return im_reg

output_list = Parallel(n_jobs=35)(
    delayed(image_reg)(
        raw_wavg_reg_bgsub_sato[0,:,:],
        raw_wavg_reg_bgsub_sato[i,:,:],
        raw_wavg_reg_bgsub[i,:,:]) 
    for i in range(nT-30)
    )
 
raw_wavg_reg_bgsub_reg = np.stack([arrays for arrays in output_list], axis=0)     

#%%

#%% Napari
   
with napari.gui_qt():
    viewer = napari.view_image(raw_wavg_reg_bgsub_reg)

#%% Saving
# io.imsave(ROOTPATH+RAWNAME[0:-4]+'_wavg.tif', raw_wavg.astype('uint8'), check_contrast=True)
# io.imsave(ROOTPATH+RAWNAME[0:-4]+'_wavg_binary.tif', raw_wavg_binary.astype('uint8')*255, check_contrast=True)
# io.imsave(ROOTPATH+RAWNAME[0:-4]+'_wavg_reg.tif', raw_wavg_reg.astype('uint8'), check_contrast=True) 
# io.imsave(ROOTPATH+RAWNAME[0:-4]+'_wavg_reg_bgsub.tif', raw_wavg_reg_bgsub.astype('float32'), check_contrast=True) 
# io.imsave(ROOTPATH+RAWNAME[0:-4]+'_wavg_reg_bgsub_edges.tif', raw_wavg_reg_bgsub_edges.astype('float32'), check_contrast=True) 
# io.imsave(ROOTPATH+RAWNAME[0:-4]+'_wavg_reg_bgsub_crop.tif', raw_wavg_reg_bgsub_crop.astype('float32'), check_contrast=True) 