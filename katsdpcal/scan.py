"""Scan class to data and operations on data."""

import numpy as np
import copy

from katcal import calprocs
from katcal.calprocs import CalSolution

#--------------------------------------------------------------------------------------------------
#--- CLASS :  Scan
#--------------------------------------------------------------------------------------------------

class Scan(object):
   
    def __init__(self, data, ti0, ti1, dump_period, antlist, bls_ordering):

        # get references to this time chunk of data
        # -- just using first polarisation for now
        self.vis = data['vis'][ti0:ti1,:,:,0] 
        self.flags = data['flags'][ti0:ti1,:,:,0] 
        self.weights = np.ones_like(self.flags,dtype=np.float)
        self.times = data['times'][ti0:ti1]
        
        # intermediate product visibility - use sparingly!
        self.modvis = None
        
        self.nchan = self.vis.shape[1]
        self.chans = range(self.nchan)
        self.nant = len(antlist)
        # baseline number includes autocorrs
        self.nbl = self.nant*(self.nant+1)/2
        
        # scan meta-data
        self.dump_period = dump_period
        self.corrprod_lookup = self.get_corrprods(antlist, bls_ordering)
        self.corr_antlists = self.get_antlists(self.corrprod_lookup)
          
    def get_corrprods(self, antlist, bls_ordering):            
        """
        Get correlation product antenna mapping
        
        Inputs:
        -------
        antlist : list of antennas, string
        bls_ordering : list of correlation products, string shape(nbl * npol,2)
        
        Returns:
        --------
        corrprod_lookup : lookup table of antenna indices for each baseline, matching data format, shape(nbl,2)
        """
        
        # make polarisation and corr_prod lookup tables (assume this doesn't change over the course of an observaton)
        antlist_index = dict([(antlist[i], i) for i in range(len(antlist))])
        corr_products_lookup = np.array([[antlist_index[a1[0:4]],antlist_index[a2[0:4]]] for a1,a2 in bls_ordering])
    
        # from full list of correlator products, get list without repeats (ie no repeats for pol)
        corrprod_lookup = -1*np.ones([self.nbl,2],dtype=np.int) # start with array of -1
        bl = -1
        for c in corr_products_lookup: 
            if not np.all(corrprod_lookup[bl] == c): 
                bl += 1
                corrprod_lookup[bl] = c
                
        return corrprod_lookup
        
    def get_antlists(self, corrprod_lookup):
        """
        Get antenna lists in solver format, from corr_prod lookup
        
        Inputs:
        -------
        corrprod_lookup : lookup table of antenna indices for each baseline, matching data format, shape(nant,2)
        
        Returns:
        --------
        antlist 1, antlist 2 : lists of antennas matching the correlation_product lookup table,
            appended with their conjugates (format required by stefcal)
        """ 
        
        # NOTE: no longer need hh and vv masks as we re-ordered the data to be ntime x nchan x nbl x npol

        # get antenna number lists for stefcal - need vis then vis.conj (assume constant over an observation)
        # assume same for hh and vv
        antlist1 = np.concatenate((corrprod_lookup[:,0], corrprod_lookup[:,1]))
        antlist2 = np.concatenate((corrprod_lookup[:,1], corrprod_lookup[:,0]))
        
        return antlist1, antlist2
        
    def solint_from_nominal(self, input_solint):   
        """
        determine appropriaye solution interval given nominal solution interval
        
        Inputs:
        ------   
        input_solint : nominal solution interval
        
        Returns:
        solint : calculated optimal solution interval
        dumps_per_solint : number of dumps per solution interval     
        """

        solint, dumps_per_solint = calprocs.solint_from_nominal(input_solint,self.dump_period,len(self.times))
        
    # ---------------------------------------------------------------------------------------------
    # Calibration solution functions
        
    def g_sol(self,input_solint,g0,REFANT,pre_apply=[]):
        
        if len(pre_apply) > 0:
            self.modvis = copy.deepcopy(self.vis)
        else:
            self.modvis = self.vis
        
        for soln in pre_apply:
           self.modvis = self.apply(soln,origvis=False) 

        # set up solution interval
        solint, dumps_per_solint = calprocs.solint_from_nominal(input_solint,self.dump_period,len(self.times))
    
        # first averge in time over solution interval
        ave_vis, ave_flags, ave_weights, av_sig, ave_times = calprocs.wavg_full_t(self.modvis,self.flags,self.weights,
                dumps_per_solint,axis=0,times=self.times)
        # second average over all channels
        ave_vis = calprocs.wavg(ave_vis,ave_flags,ave_weights,axis=1)
    
        # solve for gains G
        antlist1, antlist2 = self.corr_antlists
        g_soln = calprocs.g_fit_per_solint(ave_vis,dumps_per_solint,antlist1,antlist2,g0,REFANT)
    
        return CalSolution('G', g_soln, ave_times)
        
    def k_sol(self,chan_sample,k0,bp0,REFANT,pre_apply=[]):
        
        if len(pre_apply) > 0:
            self.modvis = copy.deepcopy(self.vis)
        else:
            self.modvis = self.vis
        
        for soln in pre_apply:
           self.modvis = self.apply(soln,origvis=False) 
    
        # average over all time (no averaging over channel)
        ave_vis = calprocs.wavg(self.modvis,self.flags,self.weights,axis=0)
    
        # solve for delay K
        antlist1, antlist2 = self.corr_antlists
        k_soln = calprocs.k_fit(ave_vis,antlist1,antlist2,self.chans,k0,bp0,REFANT,chan_sample=chan_sample)
    
        return CalSolution('K', k_soln, np.ones(len(k_soln))) 
        
    def b_sol(self,bp0,REFANT,pre_apply=[]):

        if len(pre_apply) > 0:
            self.modvis = copy.deepcopy(self.vis)
        else:
            self.modvis = self.vis
        
        for soln in pre_apply:
           self.modvis = self.apply(soln,origvis=False) 
    
        # average over all time (no averaging over channel)
        ave_vis = calprocs.wavg(self.modvis,self.flags,self.weights,axis=0)
    
        # solve for bandpass
        antlist1, antlist2 = self.corr_antlists
        b_soln = calprocs.bp_fit(ave_vis,antlist1,antlist2,bp0,REFANT)

        return CalSolution('B', b_soln, np.ones(len(b_soln))) 
        
    # ---------------------------------------------------------------------------------------------
    # solution application
    
    def _apply(self, solval, origvis=True, inplace=False):
        """
        Applies calibration solutions.
        Must already be interpolated to either full time or full frequency.
        
        Inputs:
        ------
        solval : multiplicative solution values to be applied to visibility data
        """  
        
        if inplace is True:
            self._apply_inplace(solval)
            return
        else:
            return self._apply_newvis(solval, origvis=origvis)
     
      
    def _apply_newvis(self, solval, origvis=True):
        """
        Applies calibration solutions.
        Must already be interpolated to either full time or full frequency.
        
        Inputs:
        ------
        solval : multiplicative solution values to be applied to visibility data
        """    
        
        outvis = copy.deepcopy(self.vis) if origvis else copy.deepcopy(self.modvis)
        
        for cp in range(len(self.corrprod_lookup)):
            if len(solval.shape) < 3:
                outvis[:,:,cp] /= np.expand_dims(solval[...,self.corrprod_lookup[cp][0]]*(solval[...,self.corrprod_lookup[cp][1]].conj()),axis=1)
            else:
                outvis[:,:,cp] /= solval[...,self.corrprod_lookup[cp][0]]*(solval[...,self.corrprod_lookup[cp][1]].conj())
                
        return outvis
        
    def _apply_inplace(self, solval):
        """
        Applies calibration solutions.
        Must already be interpolated to either full time or full frequency.
        
        Inputs:
        ------
        solval : multiplicative solution values to be applied to visibility data
        """    
        
        for cp in range(len(self.corrprod_lookup)):
            if len(solval.shape) < 3:
                self.vis[:,:,cp] /= np.expand_dims(solval[...,self.corrprod_lookup[cp][0]]*(solval[...,self.corrprod_lookup[cp][1]].conj()),axis=1)
            else:
                self.vis[:,:,cp] /= solval[...,self.corrprod_lookup[cp][0]]*(solval[...,self.corrprod_lookup[cp][1]].conj())
    
    def apply(self, soln, origvis=True, inplace=False):
        # set up more complex interpolation methods later
        if soln.soltype is 'G': 
            return self._apply(soln.values,origvis=origvis,inplace=inplace)    
        elif soln.soltype is 'K': 
            # want dimensions ntime x nchan x nant
            g_from_k = np.zeros([self.vis.shape[0],self.vis.shape[1],soln.values.shape[-1]],dtype=np.complex)
            for c in self.chans:
                g_from_k[:,c,:] = np.exp(1.0j*soln.values*c)
            return self._apply(g_from_k,origvis=origvis,inplace=inplace)
        elif soln.soltype is 'B': 
            return self._apply(soln.values,origvis=origvis,inplace=inplace)
        else:
            return ValueError('Solution type is invalid.')
    
    # ---------------------------------------------------------------------------------------------
    # interpolation
        
    def interpolate(self, solns):
        # set up more complex interpolation methods later
        soltype = solns.soltype
        if soltype is 'G': 
            #return self.self_interpolate(num_dumps, **kwargs)   
            return self.linear_interpolate(solns)         
        if soltype is 'K': 
            return self.inf_interpolate(solns)
        if soltype is 'B': 
            return self.inf_interpolate(solns)
            
    def linear_interpolate(self, solns):
        values = solns.values
        times = solns.times
        # interpolate complex solutions separately in real and imaginary
        real_interp = np.array([np.interp(self.times, times, v.real) for v in values.T])
        imag_interp = np.array([np.interp(self.times, times, v.imag) for v in values.T])
        interp_solns = real_interp.T + 1.0j*imag_interp.T
        return CalSolution(solns.soltype, interp_solns, self.times)
        
    def inf_interpolate(self, solns):
        values = solns.values
        interp_solns = np.repeat(np.expand_dims(values,axis=0),len(self.times),axis=0)
        return CalSolution(solns.soltype, interp_solns, self.times)
        
        
        

