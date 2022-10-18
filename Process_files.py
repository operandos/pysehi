#For creating stacks and folders in processed data
import shutil
import os
import numpy as np
import glob
import tifffile as tf
import csv
import regex
from IPython import get_ipython
import cv2
from skimage import transform
from skimage import img_as_uint
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib_scalebar.scalebar import ScaleBar
import xlsxwriter
from math import log10, floor
from read_roi import read_roi_file as rrf
import pandas as pd
import json


get_ipython().run_line_magic('matplotlib', 'inline') #change backend to plot figures in plots window

#Create folders from rawData tree, aka. source (SRC)
def ig_f(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]
def round_sig(x, sig=2):
    return round(x, sig-int(floor(log10(abs(x))))-1)
def imgDose(file):
    #Returns the electron dose per image expressed through an overall charge exposure, 
    #given by the product of the beam current I_0 and the total exposure time t_total per area A,
    #whereby typical quoted units are coulombs per metre squared (Cm^-2)
    with tf.TiffFile(file) as tif:
        fileMetadata = tif.fei_metadata #read FEI metadata
    I_0 = fileMetadata['EBeam']['BeamCurrent']
    t_dwell = fileMetadata['Scan']['Dwelltime']
    n_px = (fileMetadata['Image']['ResolutionY']/fileMetadata['EScan']['ScanInterlacing'])*fileMetadata['Image']['ResolutionX']
    n_int = fileMetadata['Scan']['Integrate']
    A = fileMetadata['Image']['ResolutionY']*pixelWidthSI*fileMetadata['Image']['ResolutionX']*pixelWidthSI
    return (I_0*t_dwell*n_px*n_int)/A

def NovaEV(destination,subdirectory,spectrumNo):
    global defV
    logdf = pd.read_csv(rf"{destination}\\{subdirectory}\\{spectrumNo}-Log.csv")
    defV = np.array(logdf['Spectrum name'][1:], float)
    deflectorStep = defV[1]
    deflectorFinal = defV[-1]
    return deflectorStep, deflectorFinal, float(deflectorStep)/2.84

def HeliosEV(sortedFiles):
    """makes list of energies (energiesEV) as global variable and returns energy step
    """
    global energiesEV
    global MVs
    global MVstep
    energiesEV = []
    MVs = []
    for file in sortedFiles:
        MV = regex.search("(?<=TLD_Mirror\d+_)[-+]?\d*([.](?!tif)\d*)*", file)
        if MV is None:
            MV = regex.search("(?<=TLD_Mirror_\d+_)[-+]?\d*([.](?!tif)\d*)*", file)
        MV = float(MV.group(0))
        MVs.append(MV)
        energiesEV.append((MV*-0.39270650395520923)+6)
        #energiesEV.append((MV*-0.3719)+6)
    MVstep = round((MVs[0] - MVs[1]),1)
    return float(energiesEV[1]-energiesEV[0])
def sortKey1(file):
    m = regex.search("(?<=TLD_Mirror\d+_)[-+]?\d*([.](?!tif)\d*)*", file) # searches for the mirrorV
    if m is None:
        m = regex.search("(?<=TLD_Mirror_\d+_)[-+]?\d*([.](?!tif)\d*)*", file) # searches for the mirrorV in oxford data
    return float(m.group())
def sortKey2(file):
    m = regex.search("\d+", file)
    return int(m.group())
def searchArea(file, area=0.8):
    "Returns corners of box y1,y2,x1,x2 which is a central portion of the original image"
    img = tf.imread(file)
    with tf.TiffFile(file) as tif:
            width, height = tif.fei_metadata['Image']['ResolutionX'], tif.fei_metadata['Image']['ResolutionY']
    
    roiHeight = int(area*height)
    roiWidth = int(area*width)
    
    return int((height - roiHeight)/2), int((height - roiHeight)/2) + roiHeight, int((width - roiWidth)/2), int((width - roiWidth)/2) + roiWidth #yi,yii,xi,xii

def finalImg(sortedFiles):
    with tf.TiffFile(file) as tif:
        resX, resY = tif.fei_metadata['Image']['ResolutionX'], tif.fei_metadata['Image']['ResolutionY']
    return tf.imread(sortedFiles[-1])[0:resY,0:resX]

SRC = input("paste raw-data folder path: ")
SRC = SRC.replace('"','')
DES = SRC.replace("Raw", "Processed")
"""Find metadata from the file SRC"""
material = SRC.split('Raw\\')[1].split('\\')[0]
matSubClass = SRC.split('Raw\\')[1].split('\\')[-1]

date = regex.search("(\d{6})|(\d*-[\d-]*\d)", SRC).group(0)

print("processed folder destination is: " + DES)

shutil.copytree(SRC, DES, ignore=ig_f)

#initialise workbook and chart
workbook = xlsxwriter.Workbook(rf'{DES}\{material}_{matSubClass}_specOut.xlsx')
worksheetSpec = workbook.add_worksheet('FOV-Spectra')
worksheetMeta = workbook.add_worksheet('Metadata')
chartNorm = workbook.add_chart({'type':'scatter','subtype':'straight'})
chartNorm.set_title ({'name': rf'{material}_{matSubClass}_norm',
                  'name_font':{'size':12}
                  })
chartNorm.set_x_axis({'name':'Energy / eV'})
chartNorm.set_y_axis({
    'name':'Emission intensity norm',
    'major_gridlines': {'visible': False}
})
chartNorm.set_legend({'position': 'top'})

chartIntensity = workbook.add_chart({'type':'scatter','subtype':'straight'})
chartIntensity.set_title ({'name': rf'{material}_{matSubClass}',
                  'name_font':{'size':12}
                  })
chartIntensity.set_x_axis({'name':'Energy / eV'})
chartIntensity.set_y_axis({
    'name':'Emission intensity au.',
    'major_gridlines': {'visible': False}
})
chartIntensity.set_legend({'position': 'top'})

countNo = 0

#initialise matching params.
noPt = 1

#Stack scaled images in data folders, write to processed folders
SRClist = os.listdir(SRC)
sorted(SRClist)
#SRClist.sort(key=lambda f: int(''.join(filter(str.isdigit, f) or -1)))
for subDir in SRClist:
    folder = SRC + "\\" + subDir
    files = glob.glob(os.path.join(folder, "*.tif")) #open image files to be used in the rest of the program
    
    #screen out non-spectrum raw data folders
    if "img" in subDir or "Img" in subDir or "im" in subDir or "Im" in subDir or "imgs" in subDir or "Imgs" in subDir:
        os.makedirs(f"{DES}\\{subDir}\\Scaled-for-Fiji")
        for file in files:
            #shutil.copy(file, DES + "\\" + subDir + "\\" + os.path.split(file)[1])
            with tf.TiffFile(file) as tif:
                pixelWidthSI = tif.fei_metadata['Scan']['PixelWidth'] #read FEI metadata
                ResolutionX = tif.fei_metadata['Image']['ResolutionX']
                ResolutionY = tif.fei_metadata['Image']['ResolutionY']
            pixelWidth = pixelWidthSI * 1e6
            AVG = cv2.imread(file, cv2.IMREAD_GRAYSCALE)[0:ResolutionY,0:ResolutionX]
            #AVG = cv2.cvtColor(AVG, cv2.COLOR_BGR2GRAY)
            imgName = os.path.split(file)[1]
            plt.figure()
            plt.imshow(AVG, cmap='gray')
            plt.axis('off')
            scalebar = ScaleBar(dx=pixelWidthSI, length_fraction=0.3, location='lower right', border_pad=0.5, color='w', box_color='k', box_alpha=0.5, font_properties={'size':'15'})
            plt.gca().add_artist(scalebar)
            bbox_inches = 0
            plt.savefig(f"{DES}\\{subDir}\\scaleBar_{imgName}", dpi=300, bbox_inches='tight',pad_inches=0)
            plt.show()
            tf.imwrite(f"{DES}\\{subDir}\\Scaled-for-Fiji\\Fiji_{imgName}", AVG, photometric='minisblack', imagej=True,
                             resolution=(1./pixelWidth, 1./pixelWidth), metadata={'unit': 'um', 'axes':'YX'})
        print("Identified an image folder")
    elif len(files) == 0:
        os.rename(DES + "\\" + subDir, DES + "\\" + subDir +"-empty")
        print("Identified an empty folder")
    elif '.tif' in subDir or '.py' in subDir:
        print('not a folder')
    else:
        #determine whether Nova or Helios
        with tf.TiffFile(files[0]) as tif:
            typeMeta = tif.fei_metadata
        SEM = typeMeta['System']['SystemType']
        if SEM == 'Nova NanoSEM 450':
            switch = 'Nova'
            bitDepth = 'uint16'
        else:
            switch = 'Helios'
            bitDepth = 'uint8'
        #sort the files
        if switch == 'Nova':
            files.sort(key=lambda f: int(''.join(filter(str.isdigit, f) or -1))) #sort img sequence into logical order
            sortedFiles = files
        else:
            sortedFiles = []
            for file in sorted(files, key = sortKey1, reverse=True):
                sortedFiles.append(file)
        #make sub folders
        os.mkdir(f"{DES}\\{subDir}\\Metadata") # for metadata outputs
        os.mkdir(f"{DES}\\{subDir}\\Colour-out") # for colouring outputs
        #set image stack XY scale from pixel resolution
        specNo = subDir#.split("_")[0]
        with tf.TiffFile(sortedFiles[0]) as tif:
            pixelWidthSI = tif.fei_metadata['Scan']['PixelWidth'] #read FEI metadata
        pixelWidth = pixelWidthSI * 1e6
        
        #get Z-axis energies
        if switch == 'Nova':
            shutil.copy(rf"{folder}\\Log.csv", DES + "\\" + subDir + "\\" + specNo + "-Log.csv") #copy-paste log file over
            deflectorStep, deflectorFinal, energyStep = NovaEV(DES,subDir,specNo)
        else:
            energyStep = HeliosEV(sortedFiles)
        
        yi,yii,xi,xii = searchArea(sortedFiles[-1], area=0.5) 
        
        i=0
        EmissionCurrents = []
        BeamCurrents = []
        Times = []
        ChPressures = []
        D_imgs = [] #image dose / Cm^-2
        labels = []
        for file in sortedFiles:
            labels.append(os.path.split(file)[1])
            with tf.TiffFile(file) as tif:
                fileMetadata = tif.fei_metadata #read FEI metadata
            if switch == 'Nova':
                EmissionCurrents.append(fileMetadata['EBeam']['EmissionCurrent'])
            else:
                BeamCurrents.append(fileMetadata['EBeam']['BeamCurrent'])
            Times.append(fileMetadata['User']['Time'])
            ChPressures.append(fileMetadata['Vacuum']['ChPressure'])
            D_img = imgDose(file)
            D_imgs.append(D_img)
            i +=1
            Img = tf.imread(file)
            Img = Img[yi:yii, xi:xii] #find features in central 80% portion of image
            Img = Img.astype(np.uint8)
        
        dataArray = np.array(tf.imread(sortedFiles)) #add img to numpy array
        tf.imwrite(rf"{DES}\\{subDir}\\Metadata\\{specNo}-stack.tif", 
                     dataArray, dtype=bitDepth, photometric='minisblack', imagej=True, 
                     resolution=(1./pixelWidth, 1./pixelWidth), metadata={'spacing':energyStep, 'unit': 'um', 'axes':'ZYX', 'Labels':labels}) #make numpy array into multi page OME-TIF format (Bio - formats)
        
        with tf.TiffFile(sortedFiles[0]) as tif:
            for page in tif.pages:
                for tag in page.tags:
                    tag_name, tag_value = tag.name, tag.value

        metadata = open(rf"{DES}\\{subDir}\\Metadata\\{specNo}-metadata.txt", 'w')
        metadata.write(tag_value)
        metadata.close()        

        if switch == 'Nova':
            emissionAverage = sum(EmissionCurrents)/len(EmissionCurrents)
            #if len(regex.findall(".+(?=[e])",str(emissionAverage).replace('.',''))[0]) >2:
            #    emissionAverage = round_sig(emissionAverage, 2)
            emissionAverage = "{:.2e}".format(emissionAverage) # returns scientific notation with 2dp
            emissionDifference = max(EmissionCurrents) - min(EmissionCurrents)
            #if len(regex.findall(".+(?=[e])",str(emissionDifference).replace('.',''))[0]) >4:
            #    emissionDifference = round_sig(emissionDifference, 4)
            emissionDifference = "{:.4e}".format(emissionDifference)
        else:
            beamAverage = sum(BeamCurrents)/len(BeamCurrents)
            if 'e' in str(beamAverage):
                beamAverage = round_sig(beamAverage, 2)
            else:
                beamAverage = round(beamAverage, 2)
            beamDifference = max(BeamCurrents) - min(BeamCurrents)
            if 'e' in str(beamDifference):
                beamDifference = round_sig(beamDifference, 4)
            else:
                beamDifference = round(beamDifference, 4)
        ChPressureAverage  = sum(ChPressures)/len(ChPressures)
        #ChPressureAverage  = round_sig(ChPressureAverage, 3)
        ChPressureAverage  = "{:.2e}".format(ChPressureAverage)
        ChPressureDifference = max(ChPressures) - min(ChPressures)
        if ChPressureDifference != 0:
            ChPressureDifference = round_sig(ChPressureDifference, 4)
                
        D_spec = sum(D_imgs) # spectrum dose        
        
        #ptsData = np.delete(ptsData, obj=0, axis=0)
        #if len(ptsData) <= 8:
        x1, y1, x2, y2 = xi, yi, xii, yii
        #else:
        #    x1, y1 = np.amin(ptsData, axis=0)
        #    x2, y2 = np.amax(ptsData, axis=0)
        #if x1 == x2 or y1 == y2:
        #    x1, y1, x2, y2 = xi, yi, xii, yii
        #add if roi area oo small
        
        #For difficult stacks a roi can be manually defined
        if len(glob.glob(rf"{SRC}\\{subDir}\\*.roi")) == 1:
            roiFile = glob.glob(rf"{SRC}\\{subDir}\\*.roi")[0] # gets roi file from raw data subDir eg. Img-50.roi
            roiDict = rrf(roiFile)
            key0 = regex.findall(".*(?=.roi)", os.path.split(roiFile)[1])[0]
            x1 = roiDict[key0]['left']
            y1 = roiDict[key0]['top']
            x2 = x1 + roiDict[key0]['width']
            y2 = y1 + roiDict[key0]['height']
        
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        fig, ax = plt.subplots()
        ax.imshow(finalImg(sortedFiles), cmap=None)
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        #plt.scatter(ptsData[:,0], ptsData[:,1])
        plt.title(subDir + " ROI for template match")
        plt.savefig(DES + "\\" + subDir + "\\Metadata\\" + specNo + "-templateMatchROI.png", dpi=300)
        plt.show()

        template = finalImg(sortedFiles)[y1:y2, x1:x2]
        template = template.astype(np.uint8)
        #plt.imshow(template)
        
        shiftList = []
        regImgList = []
        tformList = []
        for file in sortedFiles:
            offsetImg = tf.imread(file)
            with tf.TiffFile(file) as tif:
                resX, resY = tif.fei_metadata['Image']['ResolutionX'], tif.fei_metadata['Image']['ResolutionY']
            offsetImg = offsetImg[0:resY,0:resX]
            offsetImg8 = offsetImg.astype(np.uint8)
            result = cv2.matchTemplate(offsetImg8, template, cv2.TM_CCOEFF_NORMED)
            minV, maxV, minpt, maxpt = cv2.minMaxLoc(result)
            shiftX,shiftY = np.asarray(maxpt)-(x1,y1)
            shiftList.append([shiftX,shiftY])
            tform = transform.EuclideanTransform(translation = (shiftX,shiftY))
            tformList.append(tform)
            regImg = transform.warp(offsetImg, tform)
            if switch == 'Nova':
                regImg = img_as_uint(regImg)
            else:
                regImg = img_as_ubyte(regImg)
            regImgList.append(regImg)
            """
            plt.figure()
            plt.imshow(regImg)
            """
        #cropping the image stack based on the max&min shifts
        shiftArray = np.array(shiftList)
        xMax, yMax = shiftArray.max(axis=0)
        xMin, yMin = shiftArray.min(axis=0)
        
        regImgArray = np.array(regImgList)
        #regImgArray = np.delete(regImgArray, obj=0, axis=0)
        regImgArray = regImgArray[:,0-yMin:resY-yMax,0-xMin:resX-xMax]
        
        stackMeta = {}
        #for i, (file, tform) in enumerate(zip(sortedFiles, tformList)):
        for i, (file,shift) in enumerate(zip(sortedFiles,shiftList)):
            stackMeta[f'img{i+1}'] = {}
            with tf.TiffFile(file) as tif:
                stackMeta[f'img{i+1}'] = tif.fei_metadata
                stackMeta[f'img{i+1}']['Processing'] = {}
                stackMeta[f'img{i+1}']['Processing']['file'] = file
                stackMeta[f'img{i+1}']['Processing']['transformation'] = {}
                stackMeta[f'img{i+1}']['Processing']['transformation']['x'] = float(shift[0])
                stackMeta[f'img{i+1}']['Processing']['transformation']['y'] = float(shift[1])
                stackMeta[f'img{i+1}']['Processing']['crop'] = {}
                stackMeta[f'img{i+1}']['Processing']['crop']['y0'] = float(0-yMin)
                stackMeta[f'img{i+1}']['Processing']['crop']['y1'] = float(resY-yMax)
                stackMeta[f'img{i+1}']['Processing']['crop']['x0'] = float(0-xMin)
                stackMeta[f'img{i+1}']['Processing']['crop']['x1'] = float(resX-xMax)
        metaFile = open(rf"{DES}\{subDir}\Metadata\{specNo}-stack_metadata.json", "w")
        json.dump(stackMeta, metaFile)
        metaFile.close()
        
        slices = np.arange(0, len(sortedFiles), 1)
        sliceNo=0
        ZproList = []
        energyStepList = []
        while sliceNo <= len(files)-1:
            sliceAv = np.mean(regImgArray[sliceNo,:,:])
            ZproList.append(sliceAv)
            energyStepList.append(sliceNo*energyStep)
            sliceNo +=1
        intensity = np.gradient(np.array(ZproList))
        intensityNorm = intensity/np.max(intensity)
        intensityList = list(intensity)
        intensityNormList = list(intensityNorm)
        """
        plt.figure()
        plt.scatter(slices, shiftArray[:,0], label='xShift')
        plt.scatter(slices, shiftArray[:,1], label='yShift')
        plt.legend()
        plt.title(specNo + " translations")
        plt.savefig(DES + "\\" + subDir + "\\" + specNo + "-XYshift.png", dpi=300)
        plt.show()
        """
        #######   meta plots   #######
        fig, axs = plt.subplots(4, sharex=True, gridspec_kw={'height_ratios': [3, 3, 1, 1]})
        fig.suptitle(f"""{specNo}, {date}
        {min(Times)} - {max(Times)}""", fontsize=10)
        #plt.fig(figsize=(2.25,3))
        #fig.tight_layout(pad=2.0)
        axs[0].plot(slices, shiftArray[:,0], 'o', label='xShift')
        axs[0].plot(slices, shiftArray[:,1], 'o', label='yShift')
        axs[0].legend()
        #axs[0].title.set_text("translations")#, fontsize=8)
        axs[1].plot(slices, ZproList, 'r-', label = 'Z-ax.pro.')
        axs[1].set(ylabel='slice mean')
        axs[1].legend()
        #axs[1].set_title("Z-axis profile")#, fontsize=8)
        if switch == 'Nova':
            axs[2].plot(slices,EmissionCurrents, 'k-', label = 'EmisCur.')
            axs[2].set(ylabel='Amps')
            axs[2].legend()
            #axs[2].title.set_text(f"Emission current {emissionAverage}$\pm${emissionDifference/2} A")#, fontsize=8)
        else:
            axs[2].plot(slices,BeamCurrents, 'k-', label='BeamCur.')
            axs[2].set(ylabel='Amps')
            axs[2].legend()
            #axs[2].title.set_text(f"Beam current {beamAverage}$\pm${beamDifference/2} A")#, fontsize=8)
        axs[3].plot(slices,ChPressures, 'k-', label = 'ChPres.')
        axs[3].set(xlabel='slice number', ylabel='Pa')
        axs[3].legend()
        #axs[3].title.set_text(f"Chamber Pressure {ChPressureAverage}$\pm${ChPressureDifference/2} Pa")#, fontsize=8)
        plt.savefig(DES + "\\" + subDir + "\\Metadata\\" + specNo +"_metaPlots.png", dpi=300)
        #fig.show()
        #plt.imshow(np.mean(regImgArray, axis=0))
        
        #######   average image   #######
        AVG = np.array(np.mean(regImgArray, axis=0))
        plt.figure()
        plt.imshow(AVG, cmap='gray')
        plt.axis('off')
        scalebar = ScaleBar(dx=pixelWidthSI, length_fraction=0.3, location='lower right', border_pad=0.5, color='w', box_color='k', box_alpha=0.5, font_properties={'size':'15'})
        plt.gca().add_artist(scalebar)
        bbox_inches = 0
        plt.savefig(DES + "\\" + subDir + "\\" + specNo +"_AVG-scaled.png", dpi=300, bbox_inches='tight',pad_inches=0)
        plt.show()
        """"""
        colNoSpec = countNo*4
        colNoMeta = countNo*7
        
        #do spectra output
        worksheetSpec.write(0,0+colNoSpec,subDir)
        worksheetSpec.write(0,1+colNoSpec, f"HFW = {fileMetadata['EScan']['HorFieldsize']*1e6} um")
        worksheetSpec.write(1,0+colNoSpec,'Energy / eV')
        worksheetSpec.write(1,1+colNoSpec,'Z-profile')
        worksheetSpec.write(1,2+colNoSpec,'Emission intensity')
        worksheetSpec.write(1,3+colNoSpec,'Emission intensity norm')
        if switch == 'Nova':
            for row_num, data in enumerate(energyStepList):    # Energy / eV
                worksheetSpec.write(row_num+2, 0+colNoSpec, data)
        else:
            for row_num, data in enumerate(energiesEV):    # Energy / eV
                worksheetSpec.write(row_num+2, 0+colNoSpec, data)
        for row_num, data in enumerate(ZproList):          # Z-profile
            worksheetSpec.write(row_num+2, 1+colNoSpec, data)
        for row_num, data in enumerate(intensityList):     # Emission intensity
            worksheetSpec.write(row_num+2, 2+colNoSpec, data)
        for row_num, data in enumerate(intensityNormList): # Emission intensity normalised
            worksheetSpec.write(row_num+2, 3+colNoSpec, data)
        #set up the chart series
        chartNorm.add_series({
            'name':         specNo,
            'categories':   ['FOV-Spectra', 2, 0+colNoSpec, 203, 0+colNoSpec],
            'values':       ['FOV-Spectra', 2, 3+colNoSpec, 203, 3+colNoSpec]
        })
        chartIntensity.add_series({
            'name':         specNo,
            'categories':   ['FOV-Spectra', 2, 0+colNoSpec, 203, 0+colNoSpec],
            'values':       ['FOV-Spectra', 2, 2+colNoSpec, 203, 2+colNoSpec]
        })
        worksheetSpec.insert_image(row_num+3,0+colNoSpec,fr'{DES}\{specNo}\{specNo}_AVG-scaled.png', {'x_scale':0.555, 'y_scale':0.555})
        #do metedata output
        worksheetMeta.write(1,0+colNoMeta,subDir)
        worksheetMeta.write(2,0+colNoMeta,fileMetadata['User']['Date'])
        worksheetMeta.write(2,1+colNoMeta,'Start')
        worksheetMeta.write(2,3+colNoMeta,'End')
        worksheetMeta.write(2,2+colNoMeta,f'{min(Times)}')
        worksheetMeta.write(2,4+colNoMeta,f'{max(Times)}')
        worksheetMeta.write(3,0+colNoMeta, 'HFW / um')
        worksheetMeta.write(3,2+colNoMeta, fileMetadata['EScan']['HorFieldsize']*1e6)
        worksheetMeta.write(4,0+colNoMeta,'Beam voltage / V')
        worksheetMeta.write(4,2+colNoMeta,fileMetadata['Beam']['HV'])
        #Use beam current for Helios
        if switch == 'Nova':
            worksheetMeta.write(5,0+colNoMeta,'Emission current / A')
            worksheetMeta.write(5,2+colNoMeta,f'{emissionAverage} +- {float(emissionDifference)/2}')
            worksheetMeta.write(5,3+colNoMeta,f'+-{round_sig(((float(emissionDifference)/2)/float(emissionAverage))*100,3)}%')
        else:
            worksheetMeta.write(5,0+colNoMeta,'Beam current / A')
            worksheetMeta.write(5,2+colNoMeta,f'{beamAverage} +- {beamDifference/2}')
            if beamDifference == 0:
                worksheetMeta.write(5,3+colNoMeta,f'+-{beamDifference}%')
            else:
                worksheetMeta.write(5,3+colNoMeta,f'+-{round_sig(((beamDifference/2)/beamAverage)*100,3)}%')
        #
        worksheetMeta.write(6,0+colNoMeta, 'Chamber Pressure / Pa')
        worksheetMeta.write(6,2+colNoMeta,f'{ChPressureAverage} +- {ChPressureDifference/2}')
        worksheetMeta.write(7,0+colNoMeta, 'Stage: R,T,X,Y')
        worksheetMeta.write(7,1+colNoMeta, fileMetadata['Stage']['StageR'])
        worksheetMeta.write(7,2+colNoMeta, fileMetadata['Stage']['StageT'])
        worksheetMeta.write(7,3+colNoMeta, fileMetadata['Stage']['StageX'])
        worksheetMeta.write(7,4+colNoMeta, fileMetadata['Stage']['StageY'])
        worksheetMeta.write(8,0+colNoMeta, 'Working distance / mm')
        worksheetMeta.write(8,2+colNoMeta, fileMetadata['Stage']['WorkingDistance']*1e3)
        worksheetMeta.write(9,0+colNoMeta, 'D_spec / Cm^-2')
        worksheetMeta.write(9,2+colNoMeta,D_spec)
        if switch == 'Nova':
            worksheetMeta.write(10,0+colNoMeta, 'defStep')
            worksheetMeta.write(10,1+colNoMeta, deflectorStep)
            worksheetMeta.write(10,2+colNoMeta, 'defFin')
            worksheetMeta.write(10,3+colNoMeta, deflectorFinal)
        else:
            worksheetMeta.write(10,0+colNoMeta, 'MV step / V')
            worksheetMeta.write(10,2+colNoMeta, MVstep)
            worksheetMeta.write(11,0+colNoMeta, 'Suction tube / V')
            worksheetMeta.write(11,2+colNoMeta, fileMetadata['TLD']['SuctionTube'])
        ###
        worksheetMeta.write(12,0+colNoMeta, 'Time')
        for row_num, data in enumerate(Times):            # Timestamp
            worksheetMeta.write(row_num+13, 0+colNoMeta, data)
        worksheetMeta.write(12, 2+colNoMeta, 'xShift')
        for row_num, data in enumerate(shiftArray[:,0]):   # xShift
            worksheetMeta.write(row_num+13, 2+colNoMeta, data)
        worksheetMeta.write(12, 3+colNoMeta, 'yShift')
        for row_num, data in enumerate(shiftArray[:,1]):   # yShift
            worksheetMeta.write(row_num+13, 3+colNoMeta, data)
        if switch == 'Nova':
            worksheetMeta.write(12, 1+colNoMeta, 'Def V / V')
            for row_num, data in enumerate(defV):  # Nova EmissionCurrents
                worksheetMeta.write(row_num+13, 1+colNoMeta, data)
            worksheetMeta.write(12, 4+colNoMeta, 'Emis. cur.')
            for row_num, data in enumerate(EmissionCurrents):  # Nova EmissionCurrents
                worksheetMeta.write(row_num+13, 4+colNoMeta, data)
        else:
            worksheetMeta.write(12, 1+colNoMeta, 'MV')
            for row_num, data in enumerate(MVs):  # Helios BeamCurrents
                worksheetMeta.write(row_num+13, 1+colNoMeta, data)
            worksheetMeta.write(12, 4+colNoMeta, 'Beam cur.')
            for row_num, data in enumerate(BeamCurrents):  # Helios BeamCurrents
                worksheetMeta.write(row_num+13, 4+colNoMeta, data)
        worksheetMeta.write(12, 5+colNoMeta, 'ChPressure')
        for row_num, data in enumerate(ChPressures):       # ChPressures
            worksheetMeta.write(row_num+13, 5+colNoMeta, data)
        worksheetMeta.insert_image(row_num+15,0+colNoMeta, DES+"\\"+subDir+"\\Metadata\\"+specNo+"_metaPlots.png", {'x_scale':0.6149, 'y_scale':0.6105})
        """"""
        tf.imwrite(DES + "\\" + subDir + "\\" + specNo + "-average-img.tif",np.array(np.mean(regImgArray, axis=0), dtype = bitDepth), dtype=bitDepth, photometric='minisblack', imagej=True,
                     resolution=(1./pixelWidth, 1./pixelWidth), metadata={'unit': 'um', 'axes':'YX'})
        tf.imwrite(rf'{DES}\{subDir}\{specNo}-match-crop-stack.tif',
                   regImgArray, dtype=bitDepth, photometric='minisblack', imagej=True,
                   resolution=(1./pixelWidth, 1./pixelWidth), metadata={'spacing':energyStep, 'unit': 'um', 'axes':'ZYX', 'Labels':labels}) #make numpy array into multi page OME-TIF format (Bio - formats)
        print("complete!......... " + folder)
        countNo += 1
worksheetSpec.insert_chart(0,4+colNoSpec, chartNorm)
worksheetSpec.insert_chart(15,4+colNoSpec, chartIntensity)
worksheetMeta.write(0,0,SEM)
workbook.close()
print("folders processed!")