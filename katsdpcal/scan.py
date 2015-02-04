"""Scan class to data and operations on data."""

import numpy as np

#from katcal.calsolution import CalSolution

from katcal import calprocs

#--------------------------------------------------------------------------------------------------
#--- CLASS :  Scan
#--------------------------------------------------------------------------------------------------

class Scan(object):
   
    def __init__(self, data, ti0, ti1, dump_period, antlist, corr_products):

        # get references to this time chunk of data
        self.vis = data['vis'][ti0:ti1,:,:,:] 
        self.flags = data['flags'][ti0:ti1,:,:,:] 
        self.weights = np.ones_like(self.flags,dtype=np.float)
        self.times = data['times'][ti0:ti1]
        
        self.nchan = self.vis.shape[1]
        self.chans = range(self.nchan)
        self.nant = len(antlist)
        # baseline number includes autocorrs
        self.nbl = self.nant*(self.nant+1)/2
        
        # scan meta-data
        self.dump_period = dump_period
        self.corrprod_lookup = self.get_corrprods(antlist, corr_products)
        self.corr_antlists = self.get_antlists(self.corrprod_lookup)

        # kwargs may include, for example, ???
        #for key, value in kwargs.items():
        #        setattr(self, key, value)
                
          
    def get_corrprods(self, antlist, corr_products):            
        """
        Get correlation product antenna mapping
        
        Inputs:
        -------
        antlist : list of antennas, string
        corr_products : list of correlation products, string shape(nbl * npol,2)
        
        Returns:
        --------
        corrprod_lookup : lookup table of antenna indices for each baseline, matching data format, shape(nbl,2)
        """
        
        # make polarisation and corr_prod lookup tables (assume this doesn't change over the course of an observaton)
        antlist_index = dict([(antlist[i], i) for i in range(len(antlist))])
        corr_products_lookup = np.array([[antlist_index[a1[0:4]],antlist_index[a2[0:4]]] for a1,a2 in corr_products])
    
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
        
    def g_sol(self,input_solint,g0,REFANT):


        # set up solution interval
        solint, dumps_per_solint = calprocs.solint_from_nominal(input_solint,self.dump_period,len(self.times))
    
        # first averge in time over solution interval
        ave_vis, ave_flags, ave_weights, av_sig, ave_times = calprocs.wavg_full_t(self.vis,self.flags,self.weights,
                dumps_per_solint,axis=0,times=self.times)
        # second average over all channels
        ave_vis = calprocs.wavg(ave_vis,ave_flags,ave_weights,axis=1)
    
        # solve for gains G
        antlist1, antlist2 = self.corr_antlists
        g_soln = calprocs.g_fit_per_solint(ave_vis,dumps_per_solint,antlist1,antlist2,g0,REFANT)
    
        #return CalSolution('G', g_soln, ave_times, solint, corrprod_lookup)
        return CalSolution('G', g_soln, ave_times)
        
    def k_sol(self,chan_sample,k0,bp0,REFANT,pre_apply=[]):
        
        
        #for solns in pre_apply:
        #    def vis_transform(solns): 
        #        self.apply(solns)
    
    
        # average over all time (no averaging over channel)
        ave_vis = calprocs.wavg(self.vis,self.flags,self.weights,axis=0) #,transform=vis_transform)
    
        # solve for delay K
        antlist1, antlist2 = self.corr_antlists
        k_soln = calprocs.k_fit(ave_vis,antlist1,antlist2,self.chans,k0,bp0,REFANT,chan_sample=chan_sample)
    
        print len(k_soln)
    
        return CalSolution('K', k_soln, np.ones(len(k_soln))) 
        
    # ---------------------------------------------------------------------------------------------
    
      
    def _apply(self, data, solns, chans=None):
        """
        Applies calibration solutions.
        Must already be interpolated to either full time or full frequency.
   
        Parameters
        ----------
        data     : array of complex, shape(ntime x nchan x nbl)
        chans    :

        Returns
        -------
        """     
        for cp in range(len(self.corrprod_lookup)):
            if len(solns.shape) < 3:
                data[:,:,cp] /= np.expand_dims(solns[...,self.corrprod_lookup[cp][0]]*(solns[...,self.corrprod_lookup[cp][1]].conj()),axis=1)
            else:
                data[:,:,cp] /= solns[...,self.corrprod_lookup[cp][0]]*(solns[...,self.corrprod_lookup[cp][1]].conj())

        return data
    
    
    def apply(self, solns, chans=None):
        # set up more complex interpolation methods later
        if self.soltype is 'G': 
            return self._apply(solns)    
        if self.soltype is 'K': 
            # want dimensions ntime x nchan x nant
            g_from_k = np.zeros([data.shape[0],data.shape[1],self.values.shape[-1]],dtype=np.complex)
            for c in chans:
                g_from_k[:,c,:] = np.exp(1.0j*self.values*c)
            return self._apply(data, g_from_k)
        if self.soltype is 'B': 
            return self._apply(data, self.values)

        return data
    
    # ---------------------------------------------------------------------------------------------
        
    def interpolate(self, solns, **kwargs):
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
        
        
#--------------------------------------------------------------------------------------------------
#--- CLASS :  CalSolution
#--------------------------------------------------------------------------------------------------

"""
Lightweight solution class to hold calibration solutions along with their times and type.
"""

class CalSolution(object):
    def __init__(self, soltype, values, times):
        if len(values) != len(times):
            raise ValueError('Solution numbers and timestamps of unequal length!')
        
        self.soltype = soltype
        self.values = values
        # start (? middle?) times of each solution
        self.times = times
        

