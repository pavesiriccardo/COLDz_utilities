import numpy as np,cPickle

class line_candidate(object):
	def dist3d(self,pos1,pos2):
		return np.sqrt((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2+(pos1[2]-pos2[2])**2)
	def dist2d(self,pos1,pos2):
		return np.sqrt((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2)
	def __init__(self,reduint_entry):
		self.SNR=reduint_entry[0]
		self.posn=reduint_entry[1]
		if self.dist2d(self.posn,(257, 345,0))<5:
			print 'this is probably continuum'
		self.spat_templ=reduint_entry[3][0]
		self.freq_templ=reduint_entry[3][1]
		self.reduint_entry=reduint_entry
		inp=open('/data2/common/COdeep_cosmos/MFanalysis/spectra_nosmooth_pos.txt')
		for idx,line in enumerate(inp):
			splitt=line.split()
			if float(splitt[0][1:-1])==self.SNR:
				try:
					self.aper_flux=float(splitt[13])
					self.aper_maj=float(splitt[7])
					self.aper_min=float(splitt[9])
					self.aper_freq=float(splitt[11])
					self.aper_FWHM=float(splitt[15])
					self.aper_int_flux=float(splitt[17])
				except:
					print 'Aper flux reading failed'
					self.aper_flux=np.nan
					self.aper_maj=np.nan
					self.aper_min=np.nan
					self.aper_freq=np.nan
					self.aper_FWHM=np.nan
					self.aper_int_flux=np.nan
				break
		inp.close()
		inp=open('/data2/common/COdeep_cosmos/MFanalysis/spectra_nosmooth_1pix_pos.txt')
		for idx,line in enumerate(inp):
			splitt=line.split()
			if float(splitt[0][1:-1])==self.SNR:
				try:
					self.pix_flux=float(splitt[9])
					self.pix_freq=float(splitt[7])
					self.pix_FWHM=float(splitt[11])
					self.pix_int_flux=float(splitt[13])
				except:
					print '1pix flux reading failed'
					self.pix_flux=np.nan
					self.pix_freq=np.nan
					self.pix_FWHM=np.nan
					self.pix_int_flux=np.nan
				break
		inp.close()
		if self.SNR>6:
			self.purity=1
		else:
			self.assign_purity()
		self.assign_flux_corr()
		inp=open('/data2/common/COdeep_cosmos/artificial/no_smooth/post_size_881002.dat')
		posterior=cPickle.load(inp)
		inp.close()
		SNRbins=np.arange(4,7,.1)
		SNRbin_idx=np.digitize(self.SNR,SNRbins)-1
		self.size_prob=np.array([np.mean(posterior[self.spat_templ][inj_size][(SNRbin_idx-3):(SNRbin_idx+3)]) for inj_size in range(3)])
		inp=open('/data2/common/COdeep_cosmos/artificial/no_smooth/post_FWHM_343333.dat')
		posterior_freq=cPickle.load(inp)
		inp.close()
		self.FWHM_prob=np.array([np.mean(posterior_freq[self.freq_templ][inj_FWHM][(SNRbin_idx-3):(SNRbin_idx+3)]) for inj_FWHM in range(3)])
		self.completeness=self.assign_compl(self.aper_int_flux)
		self.L_prime=self.calc_L_prime(self.aper_int_flux,self.aper_freq)
		self.ra,self.dec=self.calc_ra_dec_freq(self.posn)[:2]
	def assign_purity(self):
		purity_list=[(5.7057825136246318, 0.16971273612304216), (5.6179751869231183, 0.13220971577825316), (5.5915053228705753, 0.12239661429567161), (5.5623800716677501, 0.11235070105344559), (5.5587968720192995, 0.11116754143343846), (5.558274723731115, 0.11099607738410254), (5.5561121332596297, 0.11028847879152433), (5.4947921536391204, 0.091874367030024037), (5.4349355872147616, 0.076733646723190579), (5.4348450759906912, 0.076712692443864347), (5.4177844753599951, 0.072861492988071763), (5.4126428293930298, 0.071738578845209899), (5.3387169205381069, 0.057377459245195744), (5.3294564592830218, 0.055796143006964394), (5.3224714261221155, 0.054632899274776082), (5.2856989624477784, 0.048907521167569558), (5.2838207320966157, 0.048632301455176656), (5.2766115562220657, 0.047590793850872844), (5.2725156762604097, 0.047009434940672792), (5.2582447132688817, 0.045041117227960829), (5.2533716934771348, 0.044388930230898051), (5.2519636927185127, 0.044202339244435271), (5.2157511582404386, 0.039676748061178262), (5.2142937932138125, 0.039505209240671441), (5.2072155522537891, 0.038683204765093473), (5.1781454672169689, 0.035493741268571757), (5.1771307040722077, 0.035387619732778174), (5.174122086832635, 0.035074983364520147), (5.1637486886336612, 0.034019616837431224), (5.1577183156724038, 0.033421863110228892), (5.1572901057096896, 0.033379850317390519), (5.1479074394148725, 0.032473469077311017), (5.1425717961932955, 0.031969941311481523), (5.1382673862532107, 0.031569897218705185), (5.1368489500979884, 0.031439262525525258), (5.1359714017837845, 0.03135873621929141), (5.1246396652659767, 0.030338796004167033), (5.1227820247540388, 0.030175065396551785), (5.1206791880265436, 0.029990884244358463), (5.1168762706200424, 0.029660902926660106), (5.112806470973065, 0.029312146872099657), (5.0948824293871109, 0.027828643304088331), (5.0856362510386655, 0.027095719828829183), (5.0790760223323517, 0.026588583380206643), (5.0692584616084346, 0.02584906850713195), (5.0638638341266473, 0.025452401348071394), (5.0617390954805463, 0.025298020306911927), (5.0616922304773198, 0.025294626845979856), (5.047805915104326, 0.0243109839966884), (5.0333902999584454, 0.023334552178944791), (5.0300978516165635, 0.023117714014799021), (5.0242002960009042, 0.022734899479662191), (5.0241607798895771, 0.022732358434809925), (5.0127930458066547, 0.022014372801079086), (5.0117358296785079, 0.021948898827814235), (5.0091792378334814, 0.021791467106450149), (5.0078026366247856, 0.021707221973826518), (5.0076937174959859, 0.021700571947112886), (5.0056519304776277, 0.021576332916109959), (5.0055955385057302, 0.021572912902320895)]
		#purity_list=[(5.7057825136246318, 0.56760742847628909), (5.6179751869231183, 0.33545930578775851), (5.5915053228705753, 0.57647061865297222), (5.5623800716677501, 0.20126015205126452), (5.5587968720192995, 0.55060900377809874), (5.558274723731115, 0.55019446270264949), (5.5561121332596297, 0.19786950207154241), (5.4947921536391204, 0.16694459139581075), (5.4349355872147616, 0.14062349187854578), (5.4348450759906912, 0.45235480251518728), (5.4177844753599951, 0.1337612419721515), (5.4126428293930298, 0.22890356656872168), (5.3387169205381069, 0.37944667360781253), (5.3294564592830218, 0.26245605101399588), (5.3224714261221155, 0.10075647123764195), (5.2856989624477784, 0.090163938045429645), (5.2838207320966157, 0.089652244838746176), (5.2766115562220657, 0.087713850558159417), (5.2725156762604097, 0.33268607397829064), (5.2582447132688817, 0.082955482588119217), (5.2533716934771348, 0.081735442636862166), (5.2519636927185127, 0.31888498717062103), (5.2157511582404386, 0.0728878320707773), (5.2142937932138125, 0.07256473351028081), (5.2072155522537891, 0.29012842122585453), (5.1781454672169689, 0.064991036731269561), (5.1771307040722077, 0.27183796566654611), (5.174122086832635, 0.27005623367234327), (5.1637486886336612, 0.062200083906585581), (5.1577183156724038, 0.061067327091669267), (5.1572901057096896, 0.06098769083800859), (5.1479074394148725, 0.05926898316754519), (5.1425717961932955, 0.058313676425049754), (5.1382673862532107, 0.057554463697800422), (5.1368489500979884, 0.2487081322903214), (5.1359714017837845, 0.2482217834350198), (5.1246396652659767, 0.24200907032867605), (5.1227820247540388, 0.17478966945344851), (5.1206791880265436, 0.23986726359205182), (5.1168762706200424, 0.053928985353699971), (5.112806470973065, 0.21362290086334862), (5.0948824293871109, 0.050446173385686501), (5.0856362510386655, 0.049052462081743332), (5.0790760223323517, 0.048087992558433386), (5.0692584616084346, 0.1116808356675211), (5.0638638341266473, 0.15481912874347895), (5.0617390954805463, 0.20979697883920934), (5.0616922304773198, 0.10988681508222865), (5.047805915104326, 0.20317845676334792), (5.0333902999584454, 0.041899750410453268), (5.0300978516165635, 0.19503271670194169), (5.0242002960009042, 0.040759920164905719), (5.0241607798895771, 0.040755090775121391), (5.0127930458066547, 0.18735640285086202), (5.0117358296785079, 0.039266405649784925), (5.0091792378334814, 0.038967348754239732), (5.0078026366247856, 0.17244993610323639), (5.0076937174959859, 0.097909780948133143), (5.0056519304776277, 0.18426943444126045), (5.0055955385057302, 0.038552237399528337)]		
		temp_pur=[pur[1] for pur in purity_list if pur[0]==self.SNR] #np.absolute(pur[0]-self.SNR)<1e-5]
		if len(temp_pur)>0:
			self.purity=temp_pur[0]
		else:
			self.purity=0
	def assign_flux_corr(self):
		if self.SNR>6:
			self.flux_correct=1.
			self.flux_correct_range=(1.,1.)
			from scipy.stats import norm
			self.bestf=norm(loc=1.,scale=1e-8)
		else:
			lognorm_param={-1:[ 0.00618493  ,0.39608951],0:[ 0.32328394,  0.53686663],2:[ 0.17125074 , 0.54813403],4:[ 0.18224569  ,0.51624523],6:[ 0.29691822 , 0.56654581],8:[ 0.42054354,  0.61816686]}
			from scipy.stats import lognorm
			self.bestf=lognorm(s=lognorm_param[min(8,self.spat_templ)][1],scale=np.exp(lognorm_param[min(8,self.spat_templ)][0])) 
			self.flux_correct=self.bestf.median()
			self.flux_correct_range=self.bestf.interval(.68)
	def assign_compl(self,flux_to_use):
		if np.isnan(flux_to_use):
			return np.nan
		myfit=lambda f,d,f0: max(0,1-(   1./(f+d)*np.exp(-f/f0)   ))
		compl_params_fit=np.array([[[ 0.51093145,  0.02771134],[ 0.41731388,  0.03699222],[ 0.48504448,  0.04833295]],       [[ 0.52961387,  0.06536336],[ 0.47647305,  0.09091878],[ 0.43582843,  0.11752236]],       [[ 0.48816207,  0.1055706 ],[ 0.41942854,  0.15440935],        [ 0.43130629,  0.21278495]]])
		comple=0
		for spat_bin in [0,1,2]:
			for freq_bin in [0,1,2]:
				param=compl_params_fit[spat_bin,freq_bin]
				comple+=self.size_prob[spat_bin]*self.FWHM_prob[freq_bin]*myfit(flux_to_use,*param)
		return max(.1,comple)
	def calc_L_prime(self,sdv,nu_obs,J=1):
		z=115.27*J/nu_obs-1
		from astropy.cosmology import FlatLambdaCDM
		cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
		DL=cosmo.luminosity_distance(z).value
		return 3.25e7*sdv*DL**2/(1+z)**3/nu_obs**2      
	def completeness_withbins(self,complet_bins):
		return np.sum(np.outer(self.size_prob,self.FWHM_prob).flatten()*np.array(complet_bins))
	def calc_ra_dec_freq(self,posn):
		import pyfits,pywcs
		f=pyfits.open('/data2/common/COdeep_cosmos/singlepointings/NewCOSMOS20.fits')
		mywcs=pywcs.WCS(f[0].header)
		return mywcs.wcs_pix2sky(np.array([posn]),0)[0][:3]

