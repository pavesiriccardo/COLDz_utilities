
import pyfits, numpy as np, matplotlib.pyplot as plt,pywcs

pointings=dict()
for pnt in range(1,58):
	pointings[pnt]=pyfits.open('/data2/common/goodsN/singlepointings/GN'+str(pnt)+'.smooth.fits')


pctrs=[(750, 750),(917.42261089116289, 452.48369657494271), (806.96861768041856, 452.42720948105159), (917.27641014690346, 643.54361577204429), (806.91944976081186, 643.48717830145449), (972.55227473819843, 548.05291404475804), (862.1467847912545, 547.96840887824237), (751.74125821542486, 547.93998944819577), (1027.8765912332021, 452.59629406062078), (1027.6333576756713, 643.65611434189077), (696.56242900649295, 643.48680190600885), (1082.9577056577434, 548.19350494100092), (972.35753377858964, 739.11284658633997), (862.04908851816867, 739.02841569859129), (751.7406066611278, 739.00002124865318), (696.5145640484626, 452.42683275461894), (1027.3900902000746, 834.6959976649191), (917.13018901551754, 834.58359802467032), (806.87027498486566, 834.52721018446596), (696.61030063911267, 834.52683411992791), (641.33571740936554, 547.96765577055623), (1082.6659200634335, 739.25331390538986), (586.06054236450245, 452.48256641714437), (586.20544017218651, 643.54248660716166), (862.24445720273411, 356.90848560839879), (751.74190961058332, 356.88004120398051), (641.23934777571037, 356.90773183893964), (530.93025470155146, 548.05140780128795), (641.432110586313, 739.02766325283676), (807.01777873967694, 261.34730471967583), (696.46670576900578, 261.34692766232587), (1137.990200059244, 643.824673938302), (475.84850564656352, 643.6542323830455), (916.98391698884677, 1025.6435023421591), (972.16274537836364, 930.17274167789776), (861.9513684454613, 930.08838508692384), (586.35035818532685, 834.58246985276378), (806.82108309263072, 1025.5871641495185), (531.12369254100315, 739.11134166685088), (420.81530503608366, 739.25105652666957), (751.73995494811072, 930.06001562330835), (641.5285272455709, 930.08763330328316), (861.8536245837787, 1121.148296060634), (476.09046999248693, 834.69411736097834), (696.65818893444293, 1025.5867884161421), (586.49532664011895, 1025.6423751635773), (806.77187406519113, 1216.6670992020192), (531.31717750393295, 930.17123808266251), (751.73930307643855, 1121.1199515893752), (751.74256084660067, 165.82017749913916), (1137.6498863317202, 834.86440903243613), (1027.1467380502311, 1025.7558030183136), (916.83759401061093, 1216.7233877297674), (1192.9741551254456, 739.44981751698367), (1082.3740633877021, 930.31308538970177), (971.96790955885888, 1121.2325783366778), (861.75585693364269, 1312.2081476370672),(640.14301571,   164.84791243)]	


'''
inp=open('/data2/common/goodsN/singlepointings/stds.dat')
rr=cPickle.load(inp)
inp.close()
rr=np.array(rr)
rr=np.transpose(rr)
rtemp=np.zeros((2016,58))
rtemp[:-1,1:]=rr
rr=rtemp
rr[:,0]=1
rr[rr==0]=np.inf
'''

''''
rr=[]
f=open('/data2/common/goodsN/SNRanalysis_57w/rlist_57.txt','r')
for line in f:
	rr.append(map(float,('1 '+line).split()))

f.close()
rr=np.array(rr)
rr[rr==0]=np.inf
'''


#Need    AI/r**2  and (A/r)**2 
f=pyfits.open('/data2/common/rp_at_NRAO_nm/GNstuff/all_concat/images/GN1_concat_4MHz_LSRK_dirty_statwt_flux.fits')
sens=np.nan_to_num(f[0].data)

numerator_sig=np.memmap('numerator',dtype='float32',mode='w+',shape=(2016, 1500,1500))
denom=np.memmap('denominator',dtype='float32',mode='w+',shape=(2016, 1500,1500))


def addin_one_pointing(pnt):
	#r=rr[:,pnt]
	inv_r=1/rr[:,pnt]
	fracx=pctrs[pnt][0]-int(pctrs[pnt][0])
	fracy=pctrs[pnt][1]-int(pctrs[pnt][1])
	shiftx=1.-fracx
	shifty=1.-fracy
	x150=int((pctrs[pnt][0]))-150
	y150=int((pctrs[pnt][1]))-150
	PB=np.moveaxis(sens,0,2)
	gn=np.moveaxis(pointings[pnt][0].data[:],0,2)
	#gn=np.moveaxis(pointings[pnt][0].data[0],0,2)  
	#Nchan=pointings[pnt][0].shape[1]
	Nchan=pointings[pnt][0].shape[0] 
	PBregridx=np.array([[x+(PB[j,i+1]-x)*shiftx for i,x in enumerate(PB[j]) if i<300-1] for j in range(300)])
	PBregridy=np.pad(np.array([[x+(PBregridx[j+1,i]-x)*shifty for i,x in enumerate(PBregridx[j])] for j in range(299)]),((0,0),(0,0),(0,1)),'constant')
	imgregridx=np.array([[x+(gn[j,i+1]-x)*shiftx for i,x in enumerate(gn[j]) if i<300-1] for j in range(300)])
	imgregridy=np.pad(np.array([[x+(imgregridx[j+1,i]-x)*shifty for i,x in enumerate(imgregridx[j])] for j in range(299)]),((0,0),(0,0),(0,2016-Nchan)),'constant')
	#num_sig_piece=np.moveaxis(imgregridy*PBregridy,2,0)
	num_sig_piece=np.moveaxis(np.einsum('a,bca->bca',inv_r**2,imgregridy*PBregridy),2,0)
	#num_noi_piece=np.moveaxis((np.einsum('a,bca->bca',r,PBregridy))**2,2,0)
	#denom_piece=np.moveaxis(PBregridy**2,2,0)
	denom_piece=np.moveaxis(np.einsum('a,bca->bca',inv_r,PBregridy)**2,2,0)
	#Aregrid=np.moveaxis(np.pad(PBregridy,((y150,1500-299-y150),(x150,1500-299-x150),(0,1)),'constant'),2,0)
	#imgregrid=np.moveaxis(np.pad(imgregridy,((y150,1500-299-y150),(x150,1500-299-x150),(0,2016-Nchan)),'constant'),2,0)
	#numerator_sig+=np.pad(num_sig_piece,((y150,1500-299-y150),(x150,1500-299-x150),(0,0)),'constant')
	#numerator_noi+=np.pad(num_noi_piece,((y150,1500-299-y150),(x150,1500-299-x150),(0,0)),'constant')
	#denom+=np.pad(denom_piece,((y150,1500-299-y150),(x150,1500-299-x150),(0,0)),'constant')
	numerator_sig[:,y150:(y150+299),x150:(x150+299)]+=num_sig_piece
	#numerator_noi[:,y150:(y150+299),x150:(x150+299)]+=num_noi_piece
	denom[:,y150:(y150+299),x150:(x150+299)]+=denom_piece
	print 'finished pointing',str(pnt)


#from multiprocessing import Pool
#mypool=Pool(6)
#mypool.map(addin_one_pointing,range(1,9))
#mypool.terminate()

#set14=range(29,43)
#notset14=range(1,29)+range(43,58)


numerator_sig_mem=np.memmap('numerator',dtype='float32',mode='r',shape=(2016, 1500,1500))
denom_mem=np.memmap('denominator',dtype='float32',mode='r',shape=(2016, 1500,1500))

#numerator_sig_mem[:]=numerator_sig
#denom_mem[:]=denom
del numerator_sig,denom,sens

result_sig=np.memmap('signal_mosaic',dtype='float32',mode='w+',shape=(2016, 1500,1500))
result_noi=np.memmap('noise_mosaic',dtype='float32',mode='w+',shape=(2016, 1500,1500))

result_sig[:]=numerator_sig_mem/denom_mem
del numerator_sig_mem
#result_noi=np.sqrt(numerator_noi)/denom
result_noi[:]=1/np.sqrt(denom_mem)
#del numerator_noi
denom_mem_Nmosaic=np.memmap('../Nmosaic/denominator',dtype='float32',mode='r',shape=(2016, 1500,1500))

mask=np.zeros((2016,1500,1500),dtype=bool)
for chans in range(2016):
	mask[chans]=np.logical_not(denom_mem_Nmosaic[chans]>0.1*np.max(denom_mem_Nmosaic[chans]))

#from scipy import ndimage
#plt.plot([ndimage.label(mask[chan],structure=np.ones((3,3)))[1] for chan in range(2015)])   #to check if connected

del denom_mem_Nmosaic
#dMax=np.max(denom_mem,(1,2))
#dMax=np.einsum('a,abc->abc',dMax,np.ones((2016,1500,1500)))
import numpy.ma as ma
#mask=np.logical_not(denom_mem>0.15*dMax)
del denom_mem
masked_sig=ma.array(result_sig,mask=mask,fill_value=0)
masked_noi=ma.array(result_noi,mask=mask,fill_value=0)
result_sig[:]=masked_sig.filled()
del result_sig,result_noi,masked_sig
result_sig=np.memmap('signal_mosaic',dtype='float32',mode='r+',shape=(2016, 1500,1500))

tempdiv=np.memmap('tempdiv',dtype='float32',mode='w+',shape=(2016, 1500,1500))
tempdiv[:]=result_sig/masked_noi

todiv=np.zeros(2016)
for chans in range(2016):
	todiv[chans]=np.std(ma.array(tempdiv[chans],mask=mask[chans],fill_value=0))


result_noi=np.memmap('noise_mosaic',dtype='float32',mode='r+',shape=(2016, 1500,1500))
result_noi[:]=np.einsum('abc,a->abc',masked_noi,todiv)
SNRcube=np.memmap('SNR_mosaic',dtype='float32',mode='w+',shape=(2016, 1500,1500))
SNRcube[:]=result_sig/result_noi
SNRcube[mask]=0.
