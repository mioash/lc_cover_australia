#!/usr/bin/env python
# coding: utf-8

# In[1]:


##Import and Initialize Google Earth Engine 
import ee  
from ee import batch

# Initialize the Earth Engine object, using the authentication credentials.
ee.Initialize()


# In[2]:


#Loading required assets
# Australia's shapefile
area_shape = ee.FeatureCollection('users/mioash/aust_cd66states')

# Training and Validation datasets
fc = ee.FeatureCollection('users/mioash/Calderon_etal_Australian_land-cover/train_6c_lcaus')
fct = ee.FeatureCollection('users/mioash/Calderon_etal_Australian_land-cover/test_6c_lcaus')

# Load Landsat 5, 7 & 8
l5 = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR')
l7 = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR')
l8 = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')

# Load extra datasets
ligths = ee.ImageCollection('NOAA/DMSP-OLS/NIGHTTIME_LIGHTS')
ecoregions = ee.FeatureCollection('users/mioash/tnc_aus_ecoregions')


# In[3]:


## Functions for data pre-processing and spectral index calculation

#months to take into account
mini = 1
mfin = 12

## Fmask algorithm
def maskClouds(image):
 #// bit positions: find by raising 2 to the bit flag code 
 cloudBit = 2**5 #//32
 shadowBit = 2**3 #// 8
 snowBit = 2**4 #//16
 fillBit = 2**0 #// 1
 #// extract pixel quality band
 qa = image.select('pixel_qa')    
 #// create and apply mask
 mask = qa.bitwiseAnd(cloudBit).eq(0).And(  #// no clouds
             qa.bitwiseAnd(shadowBit).eq(0)).And( #// no cloud shadows
             qa.bitwiseAnd(snowBit).eq(0)).And(   #// no snow
             qa.bitwiseAnd(fillBit).eq(0))    #// no fill
 return image.updateMask(mask)

# Calculate and add spectral indices to every image
def add_predictors(img):
  #img = img.select(['B1','B2','B3','B4','B5','B7'])
  ndvi = img.normalizedDifference(['B4', 'B3']).rename('NDVI')
  ndwi = img.normalizedDifference(['B2', 'B4']).rename(['NDWI'])
  ndbi = img.normalizedDifference(['B5', 'B4']).rename(['NDBI'])
  wbi = img.select('B1').divide(img.select('B4')).rename(['WBI'])
  evi = img.expression(
    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
      'NIR': img.select('B4'),
      'RED': img.select('B3'),
      'BLUE': img.select('B1')
}).rename(['EVI'])
  msavi2 = img.expression(
    '1/2 * ((2*NIR+1) - ((2*NIR+1)**2 - 8*(NIR-RED))**(1/2))',{
      'NIR': img.select('B4'),
      'RED': img.select('B3')
  }).rename(['MSAVI2'])
  savi = img.expression(
    '((NIR-RED) / (RED+NIR+0.5))*(1.5)', {
      'RED': img.select('B3'),
      'NIR': img.select('B4')
  }).rename(['SAVI'])
  osavi = img.expression(
    '((NIR-RED) / (RED+NIR+0.16))', {
      'RED': img.select('B3'),
      'NIR': img.select('B4')
  }).rename(['OSAVI'])
  satvi = img.expression(
    '(((SWIR1-RED) / (SWIR1+RED+0.5))*1.5) - (SWIR2/2)',{
      'SWIR1': img.select('B5'),
      'RED': img.select('B3'),
      'SWIR2': img.select('B7')
    }).rename(['SATVI'])
  bsi = img.expression( #updated
    '((SWIR2+RED)-(NIR+BLUE))/((SWIR2+RED)+(NIR+BLUE))',{
      'SWIR2': img.select('B7'),
      'NIR': img.select('B4'),
      'RED': img.select('B3'),
      'BLUE':img.select('B1')
  }).rename(['BSI'])
  br = img.select('B1').subtract(img.select('B3')).rename(['BR'])
  bg = img.normalizedDifference(['B1', 'B2']).rename(['BG'])
  indexMax = ndvi.max(ndwi).rename('indexMax')
  return img.addBands([ndvi.float(),ndwi.float(),ndbi.float(),wbi.float(),evi.float(),msavi2.float(),savi.float(),
                       osavi.float(),satvi.float(),bsi.float(),br.float(),bg.float(),indexMax.float()])

def filtro (im,y1,y2):
  filt = im.filter(ee.Filter.calendarRange(ee.Number(y1),ee.Number(y2),'year')).filter(ee.Filter.calendarRange(ee.Number(1),ee.Number(12),'month'))
  return filt


# In[4]:


# Apply functions to L5 and L7 ImageCollection
l5 = l5.filterDate(ee.Date.fromYMD(1984,1,1), ee.Date.fromYMD(2012,12,31)).filterBounds(area_shape).map(maskClouds).select(['B1','B2','B3','B4','B5','B7']).map(add_predictors)
l7 = l7.filterDate(ee.Date.fromYMD(1999,1,1), ee.Date.fromYMD(2015,12,31)).filterBounds(area_shape).map(maskClouds).select(['B1','B2','B3','B4','B5','B7']).map(add_predictors)
l8 = l8.filterDate(ee.Date.fromYMD(2013,1,1), ee.Date.fromYMD(2017,12,31)).filterBounds(area_shape).map(maskClouds)
def rrename(imgc):
    return imgc.select(['B2','B3','B4','B5','B6','B7']).rename(['B1','B2','B3','B4','B5','B7'])
l8 = l8.map(rrename).map(add_predictors)

l5_85 = filtro(l5,1984,1988).median()
l5_90 = filtro(l5,1988,1992).median()
l5_95 = filtro(l5,1993,1997).median()
l5_00 = filtro(l5,1998,2002).median()
l5_05 = filtro(l5,2003,2007).median()
l5_10 = filtro(l5,2008,2012).median()
l7_00 = filtro(l7,1999,2002).median()
l7_05 = filtro(l7,2003,2007).median()
l7_10 = filtro(l7,2008,2012).median()
l7_15 = filtro(l7,2013,2015).median()
l8_15 = filtro(l8,2013,2017).median()

def unir (im1,im2):
    unn = ee.ImageCollection([ee.Image(im1), ee.Image(im2)])
    unn = unn.reduce(ee.Reducer.mean()).select(
        ['B1_mean','B2_mean', 'B3_mean', 'B4_mean','B5_mean','B7_mean' ,'NDVI_mean','NDWI_mean','NDBI_mean','WBI_mean','EVI_mean',
         'MSAVI2_mean','SAVI_mean','OSAVI_mean','SATVI_mean','BSI_mean',
         'BR_mean','BG_mean','indexMax_mean'], #// old names
        ['B1','B2','B3','B4','B5','B7','NDVI','NDWI','NDBI','WBI','EVI',
         'MSAVI2','SAVI','OSAVI','SATVI','BSI','BR','BG','indexMax'])#// new names
    return unn

l57_00 = unir(l5_00,l7_00)
l57_05 = unir(l5_05,l7_05)
l57_10 = unir(l5_10,l7_10)


# In[5]:


def filtro1 (im,y1,y2,mini,mfin,old_bands, season):
    filt1= im.filter(ee.Filter.calendarRange(ee.Number(y1),ee.Number(y2),'year')).filter(ee.Filter.calendarRange(ee.Number(mini),ee.Number(mfin),'month'))
    new_bands = [x + season for x in oldbands]
    filt1 = filt1.median().select(old_bands,new_bands)
    return filt1

oldbands = ['B1','B2', 'B3', 'B4','B5','B7' ,'NDVI','NDWI','NDBI','WBI','EVI',
         'MSAVI2','SAVI','OSAVI','SATVI','BSI','BR','BG']
def add_seasons (img,y_ini,y_end):
    djf = filtro1(img,y_ini,y_end,12,2,oldbands,'_DJF')
    mam = filtro1(img,y_ini,y_end,3,5,oldbands,'_MAM')
    jja = filtro1(img,y_ini,y_end,6,8,oldbands,'_JJA')
    son = filtro1(img,y_ini,y_end,9,11,oldbands,'_SON')
    fs = filtro1(img,y_ini,y_end,1,6,oldbands,'_FS')
    ss = filtro1(img,y_ini,y_end,7,12,oldbands,'_SS')
    return djf.addBands(mam).addBands(jja).addBands(son).addBands(fs).addBands(ss)

l5_85a = l5_85.addBands(add_seasons(l5,1984,1988))
l5_90a = l5_90.addBands(add_seasons(l5,1988,1992))
l5_95a = l5_95.addBands(add_seasons(l5,1993,1997))
l5_00a = l5_00.addBands(add_seasons(l5,1998,2002))
l5_05a = l5_05.addBands(add_seasons(l5,2003,2007))
l5_10a = l5_10.addBands(add_seasons(l5,2008,2012))
l7_00a = l7_00.addBands(add_seasons(l7,1999,2002))
l7_05a = l7_05.addBands(add_seasons(l7,2003,2007))
l7_10a = l7_10.addBands(add_seasons(l7,2008,2012))
l8_15a = l8_15.addBands(add_seasons(l8,2013,2017))
l57_00a = l5_00a.add(l7_00a)
l57_00a = l57_00a.divide(2)

l57_05a = l5_05a.add(l7_05a)
l57_05a = l57_05a.divide(2)

l57_10a = l5_10a.add(l7_10a)
l57_10a = l57_10a.divide(2)



# In[6]:


#Adding biophysical information

def filtrolt (im):
  #filtlt= im.filterDate(ee.Date.fromYMD(1992,1,1), ee.Date.fromYMD(1995,12,31)).min().float().clip(area_shape)
  return im.filter(ee.Filter.calendarRange(1992,2000,'year')).filter(ee.Filter.calendarRange(1,12,'month')).min().float().clip(area_shape)
    
# Resample night-ligths
n1992 = filtrolt(ligths)#.ad
n1992a = n1992.select('stable_lights').unitScale(0,63).resample('bilinear')#.divide(63).resample('bilinear')
n1992a = n1992.select('stable_lights')

# Load, fill and add -bioclimatic variables, elevation and slope
proj = l5_85a.projection()
bioclim = ee.Image('WORLDCLIM/V1/BIO').select(['bio01','bio12'],['Mean_Temp', 'Prec']).clip(area_shape)
def fill_holes (img,n_iter,min_scale,max_scale,increment):
    vals1 = ee.List.sequence(min_scale,max_scale,increment)
    for i in range(0,n_iter):
        val = vals1.get(ee.Number(i))
        imm1 = img.reproject(proj.atScale(val))
        img = img.unmask(imm1,False)        
    return img.clip(area_shape)
bioclim = fill_holes(bioclim,7,2000,10000,1000)

#//Elevation
elevation = ee.Image('USGS/SRTMGL1_003').rename(['Elevation']).clip(area_shape)
slope = ee.Terrain.slope(elevation).rename(['Slope'])
eco_mask = ee.Image().float().paint(ecoregions, 'WWF_MHTNUM').rename('ecoregion').clip(area_shape)

l5_85a = l5_85a.addBands(n1992a).addBands(eco_mask)
l5_85a = l5_85a.addBands(l5_85a.normalizedDifference(['stable_lights', 'NDVI']).rename(['mndui'])).addBands(bioclim).addBands(elevation).addBands(slope)
l5_90a = l5_90a.addBands(n1992a).addBands(eco_mask)
l5_90a = l5_90a.addBands(l5_90a.normalizedDifference(['stable_lights', 'NDVI']).rename(['mndui'])).addBands(bioclim).addBands(elevation).addBands(slope)
l5_95a = l5_95a.addBands(n1992a).addBands(eco_mask)
l5_95a = l5_95a.addBands(l5_95a.normalizedDifference(['stable_lights', 'NDVI']).rename(['mndui'])).addBands(bioclim).addBands(elevation).addBands(slope)
l57_00a = l57_00a.addBands(n1992a).addBands(eco_mask)
l57_00a = l57_00a.addBands(l57_00a.normalizedDifference(['stable_lights', 'NDVI']).rename(['mndui'])).addBands(bioclim).addBands(elevation).addBands(slope)
l57_05a = l57_05a.addBands(n1992a).addBands(eco_mask)
l57_05a = l57_05a.addBands(l57_05a.normalizedDifference(['stable_lights', 'NDVI']).rename(['mndui'])).addBands(bioclim).addBands(elevation).addBands(slope)
l57_10a = l57_10a.addBands(n1992a).addBands(eco_mask)
l57_10a = l57_10a.addBands(l57_10a.normalizedDifference(['stable_lights', 'NDVI']).rename(['mndui'])).addBands(bioclim).addBands(elevation).addBands(slope)
l8_15a = l8_15a.addBands(n1992a).addBands(eco_mask)
l8_15a = l8_15a.addBands(l8_15a.normalizedDifference(['stable_lights', 'NDVI']).rename(['mndui'])).addBands(bioclim).addBands(elevation).addBands(slope)


# In[9]:


bands= ['SATVI_JJA', 'BSI_SON', 'BR_FS', 'B5_SS', 'B5_DJF', 'MSAVI2_SS','EVI_DJF','EVI_SON', 
    'BR_JJA', 'NDWI_JJA', 'NDBI_SON', 'B7_SS', 'BR_DJF', 'B4', 'BG_DJF', 'MSAVI2_FS', 
    'B2_SS', 'NDWI', 'WBI', 'Prec', 'WBI_JJA', 'OSAVI_DJF', 'SAVI', 'B1_MAM','B2_MAM', 
    'NDWI_FS', 'MSAVI2_SON', 'BR_MAM', 'NDBI_DJF', 'NDVI_DJF', 'NDBI_JJA', 'NDVI_SON',
    'BG_SS', 'B3', 'mndui', 'Mean_Temp', 'NDBI_MAM', 'BSI', 'EVI','B1_FS', 'BR_SON', 'Slope', 
    'B3_FS', 'Elevation', 'EVI_JJA', 'B2', 'B4_DJF','ecoregion']

rp = ee.FeatureCollection(fc)
rpt = ee.FeatureCollection(fct)

# Change the base-year image
imtrain = l57_05a.select(bands)
name_t = 'smile_aus_l5705_0604'

rpp= imtrain.reproject('EPSG: 3665')

trainingPartition = rpp.sampleRegions(rp,['Class'],tileScale=8)
num_arbol = 100

testingPartition = rpp.sampleRegions(rpt,['Class'],tileScale=8)

# Temporal variables for creating k-folds
t_C0 = trainingPartition.filterMetadata('Class','equals',ee.Number(0))
t_C1 = trainingPartition.filterMetadata('Class','equals',ee.Number(1))
t_C2 = trainingPartition.filterMetadata('Class','equals',ee.Number(2))
t_C3 = trainingPartition.filterMetadata('Class','equals',ee.Number(3))
t_C4 = trainingPartition.filterMetadata('Class','equals',ee.Number(4))
t_C5 = trainingPartition.filterMetadata('Class','equals',ee.Number(5))

def seedo(nnn):
    #get random columns for each land-cover type
    random0 = t_C0.randomColumn('random', nnn);
    random1 = t_C1.randomColumn('random', nnn);
    random2 = t_C2.randomColumn('random', nnn);
    random3 = t_C3.randomColumn('random', nnn);
    random4 = t_C4.randomColumn('random', nnn);
    random5 = t_C5.randomColumn('random', nnn);
    subsample = random0.filter(ee.Filter.lt('random', 0.8)).merge(
        random1.filter(ee.Filter.lt('random', 0.8))).merge(
        random2.filter(ee.Filter.lt('random', 0.8))).merge(
        random3.filter(ee.Filter.lt('random', 0.8))).merge(
        random4.filter(ee.Filter.lt('random', 0.8))).merge(
        random5.filter(ee.Filter.lt('random', 0.8))) 
    trainedClassifier = ee.Classifier.smileRandomForest(num_arbol).train(subsample, 'Class',bands)
    im_trained = imtrain.classify(trainedClassifier)
    return im_trained

c = ee.ImageCollection(ee.List.sequence(1,10).map(seedo))
d = c.toBands().regexpRename('^(.*)', 'b_$1')
modee = c.reduce(ee.Reducer.mode())


# Calculate entropy
total = c.reduce('count')

# Class and counts
fr = c.reduce(ee.Reducer.autoHistogram(6, 1))

# Just get the counts
p = fr.arraySlice(1, 1).divide(total)

# mask array
ct=p.neq(0)
p = p.arrayMask(ct)

# Log with custom base
def log_b(x,base):
    return x.log().divide(ee.Number(base).log())
# Entropy
H = log_b(p, 2).multiply(p).arrayReduce('sum', [0]).arrayFlatten([['ent'], ['']]).multiply(-1)

# Export image
llx = 108.76 
lly = -44 
urx = 155 
ury = -10  #australia
geometry = [[llx,lly], [llx,ury], [urx,ury], [urx,lly]]

img_exp=d

taske = ee.batch.Export.image.toAsset(img_exp,description=name_t+'_1_10_bands',assetId='users/mioash/'+name_t+'_1_10_bands',scale=30,region=geometry,maxPixels=1e13,crs='EPSG:4326')
taske.start()

