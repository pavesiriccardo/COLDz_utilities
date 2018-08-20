fwhm=2.355
import numpy as np,pyfits,cPickle
from matplotlib.patches import Ellipse
from astropy.modeling import models, fitting
fit_p = fitting.LevMarLSQFitter()
from scipy.optimize import curve_fit
from scipy.special import erf

def beam_factor(xpix,ypix,freqGHz,ratios,noise_list):
	nSamples=10000
	factor=freqGHz/60*0.5
	pctrs=pctrs=[(750, 750),(917.42261089116289, 452.48369657494271), (806.96861768041856, 452.42720948105159), (917.27641014690346, 643.54361577204429), (806.91944976081186, 643.48717830145449), (972.55227473819843, 548.05291404475804), (862.1467847912545, 547.96840887824237), (751.74125821542486, 547.93998944819577), (1027.8765912332021, 452.59629406062078), (1027.6333576756713, 643.65611434189077), (696.56242900649295, 643.48680190600885), (1082.9577056577434, 548.19350494100092), (972.35753377858964, 739.11284658633997), (862.04908851816867, 739.02841569859129), (751.7406066611278, 739.00002124865318), (696.5145640484626, 452.42683275461894), (1027.3900902000746, 834.6959976649191), (917.13018901551754, 834.58359802467032), (806.87027498486566, 834.52721018446596), (696.61030063911267, 834.52683411992791), (641.33571740936554, 547.96765577055623), (1082.6659200634335, 739.25331390538986), (586.06054236450245, 452.48256641714437), (586.20544017218651, 643.54248660716166), (862.24445720273411, 356.90848560839879), (751.74190961058332, 356.88004120398051), (641.23934777571037, 356.90773183893964), (530.93025470155146, 548.05140780128795), (641.432110586313, 739.02766325283676), (807.01777873967694, 261.34730471967583), (696.46670576900578, 261.34692766232587), (1137.990200059244, 643.824673938302), (475.84850564656352, 643.6542323830455), (916.98391698884677, 1025.6435023421591), (972.16274537836364, 930.17274167789776), (861.9513684454613, 930.08838508692384), (586.35035818532685, 834.58246985276378), (806.82108309263072, 1025.5871641495185), (531.12369254100315, 739.11134166685088), (420.81530503608366, 739.25105652666957), (751.73995494811072, 930.06001562330835), (641.5285272455709, 930.08763330328316), (861.8536245837787, 1121.148296060634), (476.09046999248693, 834.69411736097834), (696.65818893444293, 1025.5867884161421), (586.49532664011895, 1025.6423751635773), (806.77187406519113, 1216.6670992020192), (531.31717750393295, 930.17123808266251), (751.73930307643855, 1121.1199515893752), (751.74256084660067, 165.82017749913916), (1137.6498863317202, 834.86440903243613), (1027.1467380502311, 1025.7558030183136), (916.83759401061093, 1216.7233877297674), (1192.9741551254456, 739.44981751698367), (1082.3740633877021, 930.31308538970177), (971.96790955885888, 1121.2325783366778), (861.75585693364269, 1312.2081476370672),(640.14301571,   164.84791243)]	
	inverseIncrementRadius_p=float(nSamples-1)/43.
	A=[0 for i in range(58)]
	def poly(x):
		coef=[ 1.0]
		coef.append( -1.300633e-03)
		coef.append( 6.480550e-07)
		coef.append(-1.267928e-10)
		if x<=int(43.*inverseIncrementRadius_p)/inverseIncrementRadius_p:
			return coef[0]+coef[1]*x**2+coef[2]*x**4+coef[3]*x**6
		else:
			return 0
	for pnt in range(1,58):
			A[pnt]=poly(int(np.sqrt((xpix-pctrs[pnt][0])**2+(ypix-pctrs[pnt][1])**2)*factor*inverseIncrementRadius_p)/inverseIncrementRadius_p)
	A=np.array(A)/np.array(noise_list)   #because of new mosaic weighting scheme
	return np.dot((A[1:])**2,ratios)/np.sum(A**2)


rr=[]
f=open('../rlist_57.txt','r')
for line in f:
	rr.append(map(float,('1 '+line).split()))

f.close()
rr=np.array(rr)
rr[rr==0]=np.inf

fits=np.loadtxt('fit_sizes.txt')

#PLOT SPECTRUM WITH BLACK BARS
def gauss(x,mean,std,norm):
		return np.absolute(norm)*np.exp(-(x-mean)**2/(2*std**2))

def gauss2D((x,y),xmean,ymean,norm,a,b,theta):
	c=np.cos(theta*np.pi/180.)
	s=np.sin(theta*np.pi/180.)
	rot=np.array([[c,-s],[s,c]])
	A=np.dot(np.dot(rot,np.diag([1./a**2,1./b**2])),np.transpose(rot))
	fu=A[0,0]*(x-xmean)**2+2*A[0,1]*(x-xmean)*(y-ymean)+A[1,1]*(y-ymean)**2
	return norm*np.exp(-fu/2).ravel()


#NN= (np.pi*3.38*2.91)/np.log(2)/4*4
NN= (np.pi*4.1*3.2)/np.log(2)/4*4


inp=open('reduint_nosmooth_combined_pos.dat','r')
reduint_no_cont=cPickle.load(inp)
import matplotlib.transforms as mtransforms
from mpl_toolkits.axes_grid1.parasite_axes import SubplotHost
outp=open('spectra_positive_combined_nosmooth.txt','w')
signal=pyfits.open('../../GNmosaic_new.fits')


for idx,obj in enumerate(reduint_no_cont):
	coord=obj[1]
	bin=obj[3][1]
	spat_size=obj[3][0]
	try:
		cube_narrow=np.nan_to_num(signal[0].section[(coord[2]-bin/2):(coord[2]+bin/2),coord[1]-30:coord[1]+30,coord[0]-30:coord[0]+30])
	except IndexError as e:
		print >>outp,obj,'no_fit',e	
	startchan=max([0,coord[2]-25])
	endchan=min([2015,coord[2]+25])
	cube=np.nan_to_num(signal[0].section[startchan:endchan,coord[1]-30:coord[1]+30,coord[0]-30:coord[0]+30])
	#p_inits = models.Gaussian2D(10,30,30,1,1,0)
	#fit_p = fitting.LevMarLSQFitter()
	small_size=max(2*spat_size,6) #was 8
	#small_size=max(2.5*spat_size,8)
	ys, xs = np.mgrid[:small_size, :small_size]
	y,x=np.mgrid[:60,:60]
	img=np.sum(cube_narrow,0)
	try:
		try:
			resfit,covm=curve_fit(gauss2D,(xs,ys),np.array(img[30-small_size/2:30+small_size/2,30-small_size/2:30+small_size/2]).flatten(),(small_size/2,small_size/2,2e-2,small_size/2/fwhm,small_size/2/fwhm,0))
			maxI= np.absolute(fwhm*resfit[3])/2.
			chanwidth=0.002
			minI=np.absolute(fwhm*resfit[4])/2.
			angI=resfit[5]
			c=np.cos(angI*np.pi/180.)
			s=np.sin(angI*np.pi/180.)
			rot=np.array([[c,-s],[s,c]])
			A=np.dot(np.dot(rot,np.diag([1./maxI**2,1./minI**2])),np.transpose(rot))
			fu=A[0,0]*(x-resfit[0]-30+small_size/2)**2+2*A[0,1]*(x-resfit[0]-30+small_size/2)*(y-resfit[1]-30+small_size/2)+A[1,1]*(y-resfit[1]-30+small_size/2)**2
		except RuntimeError as e:
			small_size=max(1.5*2*spat_size,8*1.5)
			ys, xs = np.mgrid[:small_size, :small_size]
			resfit,covm=curve_fit(gauss2D,(xs,ys),np.array(img[30-small_size/2:30+small_size/2,30-small_size/2:30+small_size/2]).flatten(),(small_size/2,small_size/2,2e-2,small_size/2/fwhm,small_size/2/fwhm,0))
			maxI= np.absolute(fwhm*resfit[3])/2.
			chanwidth=0.002
			minI=np.absolute(fwhm*resfit[4])/2.
			angI=resfit[5]
			c=np.cos(angI*np.pi/180.)
			s=np.sin(angI*np.pi/180.)
			rot=np.array([[c,-s],[s,c]])
			A=np.dot(np.dot(rot,np.diag([1./maxI**2,1./minI**2])),np.transpose(rot))
			fu=A[0,0]*(x-resfit[0]-30+small_size/2)**2+2*A[0,1]*(x-resfit[0]-30+small_size/2)*(y-resfit[1]-30+small_size/2)+A[1,1]*(y-resfit[1]-30+small_size/2)**2
		if not np.all(covm<np.inf):
			small_size=max(1.5*2*spat_size,8*1.5)
			ys, xs = np.mgrid[:small_size, :small_size]
			resfit,covm=curve_fit(gauss2D,(xs,ys),np.array(img[30-small_size/2:30+small_size/2,30-small_size/2:30+small_size/2]).flatten(),(small_size/2,small_size/2,2e-2,small_size/2/fwhm,small_size/2/fwhm,0))
			maxI= np.absolute(fwhm*resfit[3])/2.
			chanwidth=0.002
			minI=np.absolute(fwhm*resfit[4])/2.
			angI=resfit[5]
			c=np.cos(angI*np.pi/180.)
			s=np.sin(angI*np.pi/180.)
			rot=np.array([[c,-s],[s,c]])
			A=np.dot(np.dot(rot,np.diag([1./maxI**2,1./minI**2])),np.transpose(rot))
			fu=A[0,0]*(x-resfit[0]-30+small_size/2)**2+2*A[0,1]*(x-resfit[0]-30+small_size/2)*(y-resfit[1]-30+small_size/2)+A[1,1]*(y-resfit[1]-30+small_size/2)**2
		#resfit=fit_p(p_inits,xs,ys,np.sum(cube_narrow,0))
		#resfit
		#maxI= np.absolute(fwhm*resfit.x_stddev.value)/2.
		#minI=np.absolute(fwhm*resfit.y_stddev.value)/2.
		#angI=resfit.theta.value/np.pi*180.
		#c=np.cos(angI*np.pi/180.)
		#s=np.sin(angI*np.pi/180.)
		#rot=np.array([[c,-s],[s,c]])
		#A=np.dot(np.dot(rot,np.diag([1./maxI**2,1./minI**2])),np.transpose(rot))
		#fu=A[0,0]*(xs-resfit.x_mean.value)**2+2*A[0,1]*(xs-resfit.x_mean.value)*(ys-resfit.y_mean.value)+A[1,1]*(ys-resfit.y_mean.value)**2
		cent_line=coord[2]*2*chanwidth+29.9605
		flux_factor=beam_factor(coord[0],coord[1],cent_line,[fits[2015*(pnt-1)+coord[2],5] for pnt in range(1,58)],rr[coord[2]])		
		spectrum=[np.sum(np.where(fu<1,cube[i],0))/NN*2/flux_factor  for i in range(cube.shape[0])]
		#sp_inits = models.Gaussian1D(1,len(spectrum)/2,1)
		#fitspec=fit_p(sp_inits,range(cube.shape[0]),spectrum)
		fitspec,covm_spec=curve_fit(gauss,range(cube.shape[0]),np.nan_to_num(spectrum),(coord[2]-startchan,bin/fwhm,2e-3))
		cent_line_err=(np.sqrt(covm_spec[0,0]))*2*chanwidth
		Sdeltv=fitspec[2]*np.sqrt(2*np.pi)*fitspec[1]*2*chanwidth/cent_line*3e5
		if not np.all(covm<np.inf):
			covm=np.zeros((6,6))
		print >>outp,obj,maxI,np.sqrt(covm[3,3])*fwhm/2.,minI,np.sqrt(covm[4,4])*fwhm/2.,cent_line,cent_line_err,fitspec[2]*1e3,1e3*np.sqrt(covm_spec[2,2]),fwhm*fitspec[1]*2*chanwidth/cent_line*3e5,fwhm*np.sqrt(covm_spec[1,1])*2*chanwidth/cent_line*3e5 ,Sdeltv,np.sqrt(2*np.pi)*2*chanwidth/cent_line*3e5*np.sqrt(covm_spec[1,1]/fitspec[1]**2+covm_spec[2,2]/fitspec[2]**2)*fitspec[1]*fitspec[2]
	except (RuntimeError,TypeError) as e:
		print >>outp,obj,'no_fit',e
	outp.flush()

