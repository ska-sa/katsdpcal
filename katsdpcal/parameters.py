# parameter file

def set_params():
    parms = {}
   
    # general data parameters
    #parms['cal_per_scan_plots'] = False
    #parms['cal_closing_plots'] = True
    parms['cal_bchan'] = 200
    parms['cal_echan'] = 800
    parms['cal_refant'] = 'ant1'
   
    # delay calibration
    #parms['cal_K'] = []
    #parms['cal_K_std'] = []
    parms['cal_k_solint'] = 10.0 # nominal pre-k g solution interval, seconds
    parms['cal_k_chan_sample'] = 10 # sample every 10th channel for pre-K BP soln
    parms['cal_kcross_chanave'] = 1 # number of channels to average together to kcross solution
   
    # bandpass calibration
    #parms['cal_BP'] = []
    #parms['cal_BP_std'] = []
    parms['cal_bp_solint'] = 10.0 # nominal pre-bp g solution interval, seconds
   
    # gain calibration
    #parms['cal_G'] = []
    #parms['cal_G_std'] = []
    parms['cal_g_solint'] = 10.0 # nominal g solution interval, seconds

    # General data parameters
    #parms['cal_check']         = False      # Only check script, don't execute tasks
    #parms['cal_debug']         = False      # run tasks debug
    #parms['cal_fluxModel']     = "PERLEY_BUTLER_2013.csv" # Filename of flux calibrator model (in FITS)
    #parms['cal_staticflags']   = "KAT7_SRFI"              # Filename containing a list of frequencies to be flagged (in FITS)

    # User Supplied data parameters
    #parms['cal_BPCal'] = []
    #parms['cal_ACal']  = []
    #parms['cal_PCal']  = []
    #parms['cal_targets'] = []
    #parms['cal_polcal'] = []

    # Archive parameters
    #parms['cal_archRoot']      = "KAT-7 DATA" # Root of ASDM/BDF data
    #parms['cal_selBand']       = "L"   # Selected band, def = first
    #parms['cal_selConfig']     = 1     # Selected frequency config, def = first
    #parms['cal_selNIF']        = 1     # Selected number of IFs, def = first
    #parms['cal_Compress']      = False

    # Hanning
    #parms['cal_doHann']       = True        # Hanning needed for RFI?
    #parms['cal_doDescm']      = True        # Descimate Hanning?

    # Parallactic angle correction
    #parms['cal_doPACor']      = False       # Make parallactic angle correction

    # Special editing list
    #parms['cal_doEditList'] =  True         # Edit using editList?
    #parms['cal_editFG'] =      1            # Table to apply edit list to
    #parms['cal_editList'] = []

    # Editing
    #parms['cal_doClearTab']   = True        # Clear cal/edit tables
    #parms['cal_doClearGain']  = True        # Clear SN and CL tables >1
    #parms['cal_doClearFlag']  = True        # Clear FG tables > 1
    #parms['cal_doClearBP']    = True        # Clear BP tables?
    #parms['cal_doQuack']      = True        # Quack data?
    #parms['cal_doBadAnt']     = True        # Check for bad antennas?
    #parms['cal_doInitFlag']   = True        # Initial broad Frequency and time domain flagging
    #parms['cal_doChopBand']   = True        # Cut the bandpass from lowest to highest channel after special editing.
    #parms['cal_editList']     = []          # List of dictionaries    
    #parms['cal_quackBegDrop'] = 5.0/60.0    # Time to drop from start of each scan in min
    #parms['cal_quackEndDrop'] = 0.0         # Time to drop from end of each scan in min
    #parms['cal_quackReason']  = "Quack"     # Reason string
    #parms['cal_begChanFrac']  = 0.2         # Fraction of beginning channels to drop
    #parms['cal_endChanFrac']  = 0.2         # Fraction of end channels to drop
    #parms['cal_doShad']       = True        # Shadow flagging (config dependent)
    #parms['cal_shadBl']       = 18.0        # Minimum shadowing baseline (m)
    #parms['cal_doElev']       = False       # Do elevation flagging
    #parms['cal_minElev']      = 15.0        # Minimum elevation to keep.
    #parms['cal_doFD1']        = True        # Do initial frequency domain flagging
    #parms['cal_FD1widMW']     = 55          # Width of the initial FD median window
    #parms['cal_FD1maxRes']    = 5.0         # Clipping level in sigma
    #parms['cal_FD1TimeAvg']   = 1.0         # time averaging in min. for initial FD flagging
    #parms['cal_FD1baseSel']   = None        # Baseline fitting region for FD1 (updates by KAT7CorrParms)
    #parms['cal_doMednTD1']    = True        # Median editing in time domain?
    #parms['cal_mednSigma']    = 5.0         # Median sigma clipping level
    #parms['cal_mednTimeWind'] = 5.0         # Median window width in min for median flagging
    #parms['cal_mednAvgTime']  = 0.1         # Median Averaging time in min
    #parms['cal_mednAvgFreq']  = 1           # Median 1=>avg chAvg chans, 2=>avg all chan, 3=> avg chan and IFs
    #parms['cal_mednChAvg']    = 5           # Median flagger number of channels to average
    #parms['cal_doRMSAvg']    = True         # Edit calibrators by RMSAvg?
    #parms['cal_RMSAvg']      = 3.0          # AutoFlag Max RMS/Avg for time domain RMS filtering
    #parms['cal_RMSTimeAvg']  = 0.1          # AutoFlag time averaging in min.
    #parms['cal_doAutoFlag']  = True         # Autoflag editing after first pass calibration?
    #parms['cal_doAutoFlag2'] = True         # Autoflag editing after final (2nd) calibration?
    #parms['cal_IClip']       = None         # AutoFlag Stokes I clipping
    #parms['cal_VClip']       = None         #
    #parms['cal_XClip']       = None         # AutoFlag cross-pol clipping
    #parms['cal_timeAvg']     = 0.33         # AutoFlag time averaging in min.
    #parms['cal_doAFFD']      = True         # do AutoFlag frequency domain flag
    #parms['cal_FDwidMW']     = 55           # Width of the median window
    #parms['cal_FDmaxRMS']    = [5.0,0.1]    # Channel RMS limits (Jy)
    #parms['cal_FDmaxRes']    = 5.0          # Max. residual flux in sigma
    #parms['cal_FDmaxResBL']  = 5.0          # Max. baseline residual
    #parms['cal_FDbaseSel']   = None         # Channels for baseline fit (Updated by KAT7CorrParms)
    #parms['cal_FDmaxAmp']    = None         # Maximum average amplitude (Jy)
    #parms['cal_FDmaxV']      = 2.0          # Maximum average VPol amp (Jy)
    #parms['cal_minAmp']      = 1.0e-5       # Minimum allowable amplitude
    #parms['cal_BChDrop']     = None         # number of channels to drop from start of each spectrum
                                     # NB: based on original number of channels, halved for Hanning
    #parms['cal_EChDrop']     = None         # number of channels to drop from end of each spectrum
                                     # NB: based on original number of channels, halved for Hanning

    # Construct a calibrator model from initial image??
    #parms['cal_getCalModel']  =  True


    # Delay calibration
    #parms['cal_doDelayCal']   =  True       # Determine/apply delays from contCals
    #parms['cal_delaySolInt']  =  10.0        # delay solution interval (min)
    #parms['cal_delaySmoo']    =  1.5        # Delay smoothing time (hr)
    #parms['cal_doTwo']        =  True      # Use two baseline combinations in delay cal
    #parms['cal_delayZeroPhs'] =  False      # Zero phase in Delay solutions?
    #parms['cal_delayBChan']   =  None       # first channel to use in delay solutions
    #parms['cal_delayEChan']   =  None       # highest channel to use in delay solutions

    # Bandpass Calibration?
    #parms['cal_doBPCal']    =    True       # Determine Bandpass calibration
    #parms['cal_bpBChan1']   =    300        # Low freq. channel,  initial cal
    #parms['cal_bpEChan1']   =    325        # Highest freq channel, initial cal, 0=>all
    #parms['cal_bpDoCenter1'] =   0.05       # Fraction of  channels in 1st, overrides bpBChan1, bpEChan1
    #parms['cal_bpBChan2']   =    None       # Low freq. channel for BP cal
    #parms['cal_bpEChan2']   =    None       # Highest freq channel for BP cal,  0=>all
    #parms['cal_bpChWid2']   =    1          # Number of channels in running mean BP soln
    #parms['cal_bpdoAuto']   =    False      # Use autocorrelations rather than cross?
    #parms['cal_bpsolMode']  =    'A&P'      # Band pass type 'A&P', 'P', 'P!A'
    #parms['cal_bpsolint1']  =    10.0/60.0  # BPass phase correction solution in min
    #parms['cal_bpsolint2']  =    10.0       # BPass bandpass solution in min 0.0->scan average
    #parms['cal_bpUVRange']  =    [0.0,0.0]   # uv range for bandpass cal
    #parms['cal_specIndex']  =    0.0         # Spectral index of BP Cal
    #parms['cal_doSpecPlot'] =    True       # Plot the amp. and phase across the spectrum

    # Amp/phase calibration parameters
    #parms['cal_doAmpPhaseCal'] = True
    #parms['cal_refAnt']   =       0          # Reference antenna
    #parms['cal_refAnts']  =      [0]         # List of Reference antenna for fringe fitting
    #parms['cal_solInt']   =      3.0         # solution interval (min)
    #parms['cal_ampScalar']=     False        # Ampscalar solutions?
    #parms['cal_solSmo']   =      0.0          # Smoothing interval for Amps (min)

    # Apply calibration and average?
    #parms['cal_doCalAvg'] =      True       # calibrate and average cont. calibrator data
    #parms['cal_avgClass'] =      "UVAvg"    # AIPS class of calibrated/averaged uv data
    #parms['cal_CalAvgTime'] =    10.0/60.0  # Time for averaging calibrated uv data (min)
    #parms['cal_CABIF'] =         1          # First IF to copy
    #parms['cal_CAEIF'] =         0          # Highest IF to copy
    #parms['cal_chAvg']   =       None       # No channel average
    #parms['cal_avgFreq'] =       None       # No channel average
    #parms['cal_doAuto']  =       True       # Export the AutoCorrs as well.

    # Right-Left delay calibration
    #parms['cal_doRLDelay'] =  False             # Determine/apply R-L delays
    #parms['cal_RLDCal']    = [(None,None,None)] # Array of triplets of (name, R-L phase (deg at 1 GHz),
                                         # RM (rad/m**2)) for calibrators
    #parms['cal_rlBChan']   = 1                  # First (1-rel) channel number
    #parms['cal_rlEChan']   = 0                  # Highest channel number. 0=> high in data.
    #parms['cal_rlUVRange'] = [0.0,0.0]          # Range of baseline used in kilowavelengths, zeros=all
    #parms['cal_rlCalCode'] = '  '               # Calibrator code
    #parms['cal_rlDoCal']   = 2                  # Apply calibration table? positive=>calibrate
    #parms['cal_rlgainUse'] = 0                  # CL/SN table to apply, 0=>highest
    #parms['cal_rltimerange']= [0.0,1000.0]      # time range of data (days)
    #parms['cal_rlDoBand']  = 1                  # If > 0 apply bandpass calibration
    #parms['cal_rlBPVer']   = 0                  # BP table to apply, 0=>highest
    #parms['cal_rlflagVer'] = 2                  # FG table version to apply
    #parms['cal_rlrefAnt']  = 0                  # Reference antenna, defaults to refAnt

    # Instrumental polarization cal?
    #parms['cal_doPolCal']  =  False      # Determine instrumental polarization from PCInsCals?
    #parms['cal_PCInsCals'] = []          # instrumental poln calibrators, name or list of names
    #parms['cal_PCFixPoln'] = False       # if True, don't solve for source polarization in ins. cal
    #parms['cal_PCpmodel']  = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]  # Instrumental poln cal source poln model.
    #parms['cal_PCAvgIF']   = False       # if True, average IFs in ins. cal.
    #parms['cal_PCSolInt']  = 2.          # instrumental solution interval (min), 0=> scan average(?)
    #parms['cal_PCRefAnt']  = 0           # Reference antenna, defaults to refAnt
    #parms['cal_PCSolType'] = "    "      # solution type, "    ", "LM  "
    #parms['cal_doPol']     = False       # Apply polarization cal in subsequent calibration?
    #parms['cal_PDVer']     = 1           # Apply PD table in subsequent polarization cal?
    #parms['cal_PCChInc']   = 5             # Channel increment in instrumental polarization
    #parms['cal_PCChWid']   = 5             # Channel averaging in instrumental polarization

    # Right-Left phase (EVPA) calibration, uses same  values as Right-Left delay calibration
    #parms['cal_doRLCal']    = False      # Set RL phases from RLCal - RLDCal or RLPCal
    #parms['cal_RLPCal']     = None       # RL Calibrator source name, in None no IF based cal.
    #parms['cal_RLPhase']    = 0.0        # R-L phase of RLPCal (deg) at 1 GHz
    #parms['cal_RLRM']       = 0.0        # R-L calibrator RM (NYI)
    #parms['cal_rlChWid']    = 3          # Number of channels in running mean RL BP soln
    #parms['cal_rlsolint1']  = 10./60     # First solution interval (min), 0=> scan average
    #parms['cal_rlsolint2']  = 5.0       # Second solution interval (min)
    #parms['cal_rlCleanRad'] = None       # CLEAN radius about center or None=autoWin
    #parms['cal_rlFOV']      = 0.05       # Field of view radius (deg) needed to image RLPCal

    # Recalibration
    #parms['cal_doRecal']       = True        # Redo calibration after editing
    #parms['cal_doDelayCal2']   = True       # Group Delay calibration of averaged data?, 2nd pass
    #parms['cal_doBPCal2']      = True        # Determine Bandpass calibration, 2nd pass
    #parms['cal_doAmpPhaseCal2']= True        # Amplitude/phase calibration, 2nd pass
    #parms['cal_doAutoFlag2']   = True        # Autoflag editing after final calibration?

    # Imaging  targets
    #parms['cal_doImage']     = True         # Image targets
    #parms['cal_targets']     = []           # List of target sources
    #parms['cal_outIClass']   = "IClean"     # Output target final image class
    #parms['cal_Stokes']      = "I"          # Stokes to image
    #parms['cal_Robust']      = 0.0          # Weighting robust parameter
    #parms['cal_FOV']         = 2.0          # Field of view radius in deg.
    #parms['cal_Niter']       = 2000         # Max number of clean iterations
    #parms['cal_minFlux']     = None         # Minimum CLEAN flux density
    #parms['cal_minSNR']      = 4.0          # Minimum Allowed SNR
    #parms['cal_solPMode']    = "P"          # Phase solution for phase self cal
    #parms['cal_solPType']    = "    "       # Solution type for phase self cal
    #parms['cal_solAMode']    = "A&P"        # Delay solution for A&P self cal
    #parms['cal_solAType']    = "    "       # Solution type for A&P self cal
    #parms['cal_avgPol']      = True         # Average poln in self cal?
    #parms['cal_avgIF']       = False        # Average IF in self cal?
    #parms['cal_maxPSCLoop']  = 3            # Max. number of phase self cal loops
    #parms['cal_minFluxPSC']  = None         # Min flux density peak for phase self cal
    #parms['cal_solPInt']     = 0.2          # phase self cal solution interval (min)
    #parms['cal_maxASCLoop']  = 1            # Max. number of Amp+phase self cal loops
    #parms['cal_minFluxASC']  = None         # Min flux density peak for amp+phase self cal
    #parms['cal_solAInt']     = 1.0          # amp+phase self cal solution interval (min)
    #parms['cal_nTaper']      = 0            # Number of additional imaging multiresolution tapers
    #parms['cal_Tapers']      = [0.0]        # List of tapers in pixels
    #parms['cal_do3D']        = False         # Make ref. pixel tangent to celest. sphere for each facet
    #parms['cal_noNeg']       = False        # F=Allow negative components in self cal model
    #parms['cal_BLFact']      = 1.00         # Baseline dependent time averaging
    #parms['cal_BLchAvg']     = False        # Baseline dependent frequency averaging
    #parms['cal_doMB']        = None         # Use wideband imaging?
    #parms['cal_MBnorder']    = None         # order on wideband imaging
    #parms['cal_MBmaxFBW']    = None         # max. MB fractional bandwidth (Set by KAT7InitContFQParms)
    #parms['cal_PBCor']       = False        # Pri. beam corr on final image
    #parms['cal_antSize']     = 12.0         # ant. diameter (m) for PBCor
    #parms['cal_CleanRad']    = None         # CLEAN radius (pix?) about center or None=autoWin
    #parms['cal_xCells']      = 15.0         # x-cell size in final image
    #parms['cal_yCells']      = 15.0         # y-cell  "
    #parms['cal_nx']          = []           # x-Size of a facet in pixels
    #parms['cal_ny']          = []           # y-size of a facet in pixels
    #parms['cal_Reuse']       = 0.0          # How many CC's to reuse after each self-cal loop??
    #parms['cal_minPatch']    = 500          # Minumum beam patch to subtract in pixels
    #parms['cal_OutlierSize'] = 300          # Size of outlier fields
    #parms['cal_autoCen']     = False        # Do autoCen? 
    #parms['cal_outlierArea'] = 4            # Multiple of FOV around phase center to find outlying CC's

    # Final
    #parms['cal_doReport']  =     True       # Generate source report?
    #parms['cal_outDisk']   =     0          # FITS disk number for output (0=cwd)
    #parms['cal_doSaveUV']  =     True       # Save uv data
    #parms['cal_doSaveImg'] =     True       # Save images
    #parms['cal_doSaveTab'] =     True       # Save Tables
    #parms['cal_doCleanup'] =     True       # Destroy AIPS files

    # diagnostics
    #parms['cal_plotSource']    =  None      # Name of source for spectral plot
    #parms['cal_plotTime']      =  [0.,1000.]  # timerange for spectral plot
    #parms['cal_doRawSpecPlot'] =  True       # Plot diagnostic raw spectra?
    #parms['cal_doSpecPlot']    =  True       # Plot diagnostic spectra?
    #parms['cal_doSNPlot']      =  True       # Plot SN tables etc
    #parms['cal_doDiagPlots']   =  True       # Plot single s']ource diagnostics
    #parms['cal_doKntrPlots']   =  False      # Contour plots
    #parms['cal_doMetadata']    =  True       # Save source and project metadata
    #parms['cal_doHTML']        =  True       # Output HTML report
    #parms['cal_doVOTable']     =  True       # VOTable
   
    return parms
    
def init_ts(ts):
    # start with empty Telescope State
    try:
        for key in ts.keys(): ts.delete(key)
    except AttributeError:
        # the Telescope State is empty
        pass
    # then populate with parameters from parameter file
    param_dict = set_params()
    for key in param_dict.keys(): ts.add(key, param_dict[key])
