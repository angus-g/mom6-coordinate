import _remapping
import f90wrap.runtime
import logging

class Mom_Remapping(f90wrap.runtime.FortranModule):
    """
    Module mom_remapping
    
    
    Defined at mom6/src/ALE/MOM_remapping.F90 lines 2-1769
    
    """
    class Remapping_Cs(f90wrap.runtime.FortranDerivedType):
        """
        Type(name=remapping_cs)
        
        
        Defined at mom6/src/ALE/MOM_remapping.F90 lines 22-35
        
        """
        def __init__(self, handle=None):
            """
            self = Remapping_Cs()
            
            
            Defined at mom6/src/ALE/MOM_remapping.F90 lines 22-35
            
            
            Returns
            -------
            this : Remapping_Cs
            	Object to be constructed
            
            
            Automatically generated constructor for remapping_cs
            """
            f90wrap.runtime.FortranDerivedType.__init__(self)
            self._handle = _remapping.f90wrap_remapping_cs_initialise()
        
        def __del__(self):
            """
            Destructor for class Remapping_Cs
            
            
            Defined at mom6/src/ALE/MOM_remapping.F90 lines 22-35
            
            Parameters
            ----------
            this : Remapping_Cs
            	Object to be destructed
            
            
            Automatically generated destructor for remapping_cs
            """
            if self._alloc:
                _remapping.f90wrap_remapping_cs_finalise(this=self._handle)
        
        @property
        def remapping_scheme(self):
            """
            Element remapping_scheme ftype=integer  pytype=int
            
            
            Defined at mom6/src/ALE/MOM_remapping.F90 line 24
            
            """
            return _remapping.f90wrap_remapping_cs__get__remapping_scheme(self._handle)
        
        @remapping_scheme.setter
        def remapping_scheme(self, remapping_scheme):
            _remapping.f90wrap_remapping_cs__set__remapping_scheme(self._handle, \
                remapping_scheme)
        
        @property
        def degree(self):
            """
            Element degree ftype=integer  pytype=int
            
            
            Defined at mom6/src/ALE/MOM_remapping.F90 line 26
            
            """
            return _remapping.f90wrap_remapping_cs__get__degree(self._handle)
        
        @degree.setter
        def degree(self, degree):
            _remapping.f90wrap_remapping_cs__set__degree(self._handle, degree)
        
        @property
        def boundary_extrapolation(self):
            """
            Element boundary_extrapolation ftype=logical pytype=bool
            
            
            Defined at mom6/src/ALE/MOM_remapping.F90 line 28
            
            """
            return \
                _remapping.f90wrap_remapping_cs__get__boundary_extrapolation(self._handle)
        
        @boundary_extrapolation.setter
        def boundary_extrapolation(self, boundary_extrapolation):
            _remapping.f90wrap_remapping_cs__set__boundary_extrapolation(self._handle, \
                boundary_extrapolation)
        
        @property
        def check_reconstruction(self):
            """
            Element check_reconstruction ftype=logical pytype=bool
            
            
            Defined at mom6/src/ALE/MOM_remapping.F90 line 30
            
            """
            return _remapping.f90wrap_remapping_cs__get__check_reconstruction(self._handle)
        
        @check_reconstruction.setter
        def check_reconstruction(self, check_reconstruction):
            _remapping.f90wrap_remapping_cs__set__check_reconstruction(self._handle, \
                check_reconstruction)
        
        @property
        def check_remapping(self):
            """
            Element check_remapping ftype=logical pytype=bool
            
            
            Defined at mom6/src/ALE/MOM_remapping.F90 line 32
            
            """
            return _remapping.f90wrap_remapping_cs__get__check_remapping(self._handle)
        
        @check_remapping.setter
        def check_remapping(self, check_remapping):
            _remapping.f90wrap_remapping_cs__set__check_remapping(self._handle, \
                check_remapping)
        
        @property
        def force_bounds_in_subcell(self):
            """
            Element force_bounds_in_subcell ftype=logical pytype=bool
            
            
            Defined at mom6/src/ALE/MOM_remapping.F90 line 34
            
            """
            return \
                _remapping.f90wrap_remapping_cs__get__force_bounds_in_subcell(self._handle)
        
        @force_bounds_in_subcell.setter
        def force_bounds_in_subcell(self, force_bounds_in_subcell):
            _remapping.f90wrap_remapping_cs__set__force_bounds_in_subcell(self._handle, \
                force_bounds_in_subcell)
        
        def __str__(self):
            ret = ['<remapping_cs>{\n']
            ret.append('    remapping_scheme : ')
            ret.append(repr(self.remapping_scheme))
            ret.append(',\n    degree : ')
            ret.append(repr(self.degree))
            ret.append(',\n    boundary_extrapolation : ')
            ret.append(repr(self.boundary_extrapolation))
            ret.append(',\n    check_reconstruction : ')
            ret.append(repr(self.check_reconstruction))
            ret.append(',\n    check_remapping : ')
            ret.append(repr(self.check_remapping))
            ret.append(',\n    force_bounds_in_subcell : ')
            ret.append(repr(self.force_bounds_in_subcell))
            ret.append('}')
            return ''.join(ret)
        
        _dt_array_initialisers = []
        
    @staticmethod
    def remapping_core_h(h0, u0, h1, cs):
        """
        u1 = remapping_core_h(h0, u0, h1, cs)
        
        
        Defined at mom6/src/ALE/MOM_remapping.F90 lines 161-223
        
        Parameters
        ----------
        h0 : double array
        u0 : double array
        h1 : double array
        u1 : double array
        cs : Remapping_Cs
        
        """
        return _remapping.f90wrap_remapping_core_h(h0=h0, u0=u0, h1=h1, cs=cs._handle)
    
    @property
    def remapping_pcm(self):
        """
        Element remapping_pcm ftype=integer pytype=int
        
        
        Defined at mom6/src/ALE/MOM_remapping.F90 line 44
        
        """
        return _remapping.f90wrap_mom_remapping__get__remapping_pcm()
    
    @property
    def remapping_plm(self):
        """
        Element remapping_plm ftype=integer pytype=int
        
        
        Defined at mom6/src/ALE/MOM_remapping.F90 line 45
        
        """
        return _remapping.f90wrap_mom_remapping__get__remapping_plm()
    
    @property
    def remapping_ppm_h4(self):
        """
        Element remapping_ppm_h4 ftype=integer pytype=int
        
        
        Defined at mom6/src/ALE/MOM_remapping.F90 line 46
        
        """
        return _remapping.f90wrap_mom_remapping__get__remapping_ppm_h4()
    
    @property
    def remapping_ppm_ih4(self):
        """
        Element remapping_ppm_ih4 ftype=integer pytype=int
        
        
        Defined at mom6/src/ALE/MOM_remapping.F90 line 47
        
        """
        return _remapping.f90wrap_mom_remapping__get__remapping_ppm_ih4()
    
    @property
    def remapping_pqm_ih4ih3(self):
        """
        Element remapping_pqm_ih4ih3 ftype=integer pytype=int
        
        
        Defined at mom6/src/ALE/MOM_remapping.F90 line 48
        
        """
        return _remapping.f90wrap_mom_remapping__get__remapping_pqm_ih4ih3()
    
    @property
    def remapping_pqm_ih6ih5(self):
        """
        Element remapping_pqm_ih6ih5 ftype=integer pytype=int
        
        
        Defined at mom6/src/ALE/MOM_remapping.F90 line 49
        
        """
        return _remapping.f90wrap_mom_remapping__get__remapping_pqm_ih6ih5()
    
    @property
    def integration_pcm(self):
        """
        Element integration_pcm ftype=integer pytype=int
        
        
        Defined at mom6/src/ALE/MOM_remapping.F90 line 51
        
        """
        return _remapping.f90wrap_mom_remapping__get__integration_pcm()
    
    @property
    def integration_plm(self):
        """
        Element integration_plm ftype=integer pytype=int
        
        
        Defined at mom6/src/ALE/MOM_remapping.F90 line 52
        
        """
        return _remapping.f90wrap_mom_remapping__get__integration_plm()
    
    @property
    def integration_ppm(self):
        """
        Element integration_ppm ftype=integer pytype=int
        
        
        Defined at mom6/src/ALE/MOM_remapping.F90 line 53
        
        """
        return _remapping.f90wrap_mom_remapping__get__integration_ppm()
    
    @property
    def integration_pqm(self):
        """
        Element integration_pqm ftype=integer pytype=int
        
        
        Defined at mom6/src/ALE/MOM_remapping.F90 line 54
        
        """
        return _remapping.f90wrap_mom_remapping__get__integration_pqm()
    
    @property
    def mod(self):
        """
        Element mod ftype=character(len=40) pytype=str
        
        
        Defined at mom6/src/ALE/MOM_remapping.F90 line 56
        
        """
        return _remapping.f90wrap_mom_remapping__get__mod()
    
    @mod.setter
    def mod(self, mod):
        _remapping.f90wrap_mom_remapping__set__mod(mod)
    
    @property
    def remappingschemesdoc(self):
        """
        Element remappingschemesdoc ftype=character(len=256) pytype=str
        
        
        Defined at mom6/src/ALE/MOM_remapping.F90 line 65
        
        """
        return _remapping.f90wrap_mom_remapping__get__remappingschemesdoc()
    
    @remappingschemesdoc.setter
    def remappingschemesdoc(self, remappingschemesdoc):
        _remapping.f90wrap_mom_remapping__set__remappingschemesdoc(remappingschemesdoc)
    
    @property
    def remappingdefaultscheme(self):
        """
        Element remappingdefaultscheme ftype=character(len=3) pytype=str
        
        
        Defined at mom6/src/ALE/MOM_remapping.F90 line 66
        
        """
        return _remapping.f90wrap_mom_remapping__get__remappingdefaultscheme()
    
    @remappingdefaultscheme.setter
    def remappingdefaultscheme(self, remappingdefaultscheme):
        \
            _remapping.f90wrap_mom_remapping__set__remappingdefaultscheme(remappingdefaultscheme)
    
    @property
    def h_neglect(self):
        """
        Element h_neglect ftype=real pytype=float
        
        
        Defined at mom6/src/ALE/MOM_remapping.F90 line 73
        
        """
        return _remapping.f90wrap_mom_remapping__get__h_neglect()
    
    @property
    def old_algorithm(self):
        """
        Element old_algorithm ftype=logical pytype=bool
        
        
        Defined at mom6/src/ALE/MOM_remapping.F90 line 78
        
        """
        return _remapping.f90wrap_mom_remapping__get__old_algorithm()
    
    def __str__(self):
        ret = ['<mom_remapping>{\n']
        ret.append('    remapping_pcm : ')
        ret.append(repr(self.remapping_pcm))
        ret.append(',\n    remapping_plm : ')
        ret.append(repr(self.remapping_plm))
        ret.append(',\n    remapping_ppm_h4 : ')
        ret.append(repr(self.remapping_ppm_h4))
        ret.append(',\n    remapping_ppm_ih4 : ')
        ret.append(repr(self.remapping_ppm_ih4))
        ret.append(',\n    remapping_pqm_ih4ih3 : ')
        ret.append(repr(self.remapping_pqm_ih4ih3))
        ret.append(',\n    remapping_pqm_ih6ih5 : ')
        ret.append(repr(self.remapping_pqm_ih6ih5))
        ret.append(',\n    integration_pcm : ')
        ret.append(repr(self.integration_pcm))
        ret.append(',\n    integration_plm : ')
        ret.append(repr(self.integration_plm))
        ret.append(',\n    integration_ppm : ')
        ret.append(repr(self.integration_ppm))
        ret.append(',\n    integration_pqm : ')
        ret.append(repr(self.integration_pqm))
        ret.append(',\n    mod : ')
        ret.append(repr(self.mod))
        ret.append(',\n    remappingschemesdoc : ')
        ret.append(repr(self.remappingschemesdoc))
        ret.append(',\n    remappingdefaultscheme : ')
        ret.append(repr(self.remappingdefaultscheme))
        ret.append(',\n    h_neglect : ')
        ret.append(repr(self.h_neglect))
        ret.append(',\n    old_algorithm : ')
        ret.append(repr(self.old_algorithm))
        ret.append('}')
        return ''.join(ret)
    
    _dt_array_initialisers = []
    

mom_remapping = Mom_Remapping()

