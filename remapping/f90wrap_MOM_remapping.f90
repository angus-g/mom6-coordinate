! Module mom_remapping defined in file mom6/src/ALE/MOM_remapping.F90

subroutine f90wrap_remapping_cs__get__remapping_scheme(this, &
    f90wrap_remapping_scheme)
    use mom_remapping, only: remapping_cs
    implicit none
    type remapping_cs_ptr_type
        type(remapping_cs), pointer :: p => NULL()
    end type remapping_cs_ptr_type
    integer, intent(in)   :: this(2)
    type(remapping_cs_ptr_type) :: this_ptr
    integer, intent(out) :: f90wrap_remapping_scheme
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_remapping_scheme = this_ptr%p%remapping_scheme
end subroutine f90wrap_remapping_cs__get__remapping_scheme

subroutine f90wrap_remapping_cs__set__remapping_scheme(this, &
    f90wrap_remapping_scheme)
    use mom_remapping, only: remapping_cs
    implicit none
    type remapping_cs_ptr_type
        type(remapping_cs), pointer :: p => NULL()
    end type remapping_cs_ptr_type
    integer, intent(in)   :: this(2)
    type(remapping_cs_ptr_type) :: this_ptr
    integer, intent(in) :: f90wrap_remapping_scheme
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%remapping_scheme = f90wrap_remapping_scheme
end subroutine f90wrap_remapping_cs__set__remapping_scheme

subroutine f90wrap_remapping_cs__get__degree(this, f90wrap_degree)
    use mom_remapping, only: remapping_cs
    implicit none
    type remapping_cs_ptr_type
        type(remapping_cs), pointer :: p => NULL()
    end type remapping_cs_ptr_type
    integer, intent(in)   :: this(2)
    type(remapping_cs_ptr_type) :: this_ptr
    integer, intent(out) :: f90wrap_degree
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_degree = this_ptr%p%degree
end subroutine f90wrap_remapping_cs__get__degree

subroutine f90wrap_remapping_cs__set__degree(this, f90wrap_degree)
    use mom_remapping, only: remapping_cs
    implicit none
    type remapping_cs_ptr_type
        type(remapping_cs), pointer :: p => NULL()
    end type remapping_cs_ptr_type
    integer, intent(in)   :: this(2)
    type(remapping_cs_ptr_type) :: this_ptr
    integer, intent(in) :: f90wrap_degree
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%degree = f90wrap_degree
end subroutine f90wrap_remapping_cs__set__degree

subroutine f90wrap_remapping_cs__get__boundary_extrapolation(this, &
    f90wrap_boundary_extrapolation)
    use mom_remapping, only: remapping_cs
    implicit none
    type remapping_cs_ptr_type
        type(remapping_cs), pointer :: p => NULL()
    end type remapping_cs_ptr_type
    integer, intent(in)   :: this(2)
    type(remapping_cs_ptr_type) :: this_ptr
    logical, intent(out) :: f90wrap_boundary_extrapolation
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_boundary_extrapolation = this_ptr%p%boundary_extrapolation
end subroutine f90wrap_remapping_cs__get__boundary_extrapolation

subroutine f90wrap_remapping_cs__set__boundary_extrapolation(this, &
    f90wrap_boundary_extrapolation)
    use mom_remapping, only: remapping_cs
    implicit none
    type remapping_cs_ptr_type
        type(remapping_cs), pointer :: p => NULL()
    end type remapping_cs_ptr_type
    integer, intent(in)   :: this(2)
    type(remapping_cs_ptr_type) :: this_ptr
    logical, intent(in) :: f90wrap_boundary_extrapolation
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%boundary_extrapolation = f90wrap_boundary_extrapolation
end subroutine f90wrap_remapping_cs__set__boundary_extrapolation

subroutine f90wrap_remapping_cs__get__check_reconstruction(this, &
    f90wrap_check_reconstruction)
    use mom_remapping, only: remapping_cs
    implicit none
    type remapping_cs_ptr_type
        type(remapping_cs), pointer :: p => NULL()
    end type remapping_cs_ptr_type
    integer, intent(in)   :: this(2)
    type(remapping_cs_ptr_type) :: this_ptr
    logical, intent(out) :: f90wrap_check_reconstruction
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_check_reconstruction = this_ptr%p%check_reconstruction
end subroutine f90wrap_remapping_cs__get__check_reconstruction

subroutine f90wrap_remapping_cs__set__check_reconstruction(this, &
    f90wrap_check_reconstruction)
    use mom_remapping, only: remapping_cs
    implicit none
    type remapping_cs_ptr_type
        type(remapping_cs), pointer :: p => NULL()
    end type remapping_cs_ptr_type
    integer, intent(in)   :: this(2)
    type(remapping_cs_ptr_type) :: this_ptr
    logical, intent(in) :: f90wrap_check_reconstruction
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%check_reconstruction = f90wrap_check_reconstruction
end subroutine f90wrap_remapping_cs__set__check_reconstruction

subroutine f90wrap_remapping_cs__get__check_remapping(this, &
    f90wrap_check_remapping)
    use mom_remapping, only: remapping_cs
    implicit none
    type remapping_cs_ptr_type
        type(remapping_cs), pointer :: p => NULL()
    end type remapping_cs_ptr_type
    integer, intent(in)   :: this(2)
    type(remapping_cs_ptr_type) :: this_ptr
    logical, intent(out) :: f90wrap_check_remapping
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_check_remapping = this_ptr%p%check_remapping
end subroutine f90wrap_remapping_cs__get__check_remapping

subroutine f90wrap_remapping_cs__set__check_remapping(this, &
    f90wrap_check_remapping)
    use mom_remapping, only: remapping_cs
    implicit none
    type remapping_cs_ptr_type
        type(remapping_cs), pointer :: p => NULL()
    end type remapping_cs_ptr_type
    integer, intent(in)   :: this(2)
    type(remapping_cs_ptr_type) :: this_ptr
    logical, intent(in) :: f90wrap_check_remapping
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%check_remapping = f90wrap_check_remapping
end subroutine f90wrap_remapping_cs__set__check_remapping

subroutine f90wrap_remapping_cs__get__force_bounds_in_subcell(this, &
    f90wrap_force_bounds_in_subcell)
    use mom_remapping, only: remapping_cs
    implicit none
    type remapping_cs_ptr_type
        type(remapping_cs), pointer :: p => NULL()
    end type remapping_cs_ptr_type
    integer, intent(in)   :: this(2)
    type(remapping_cs_ptr_type) :: this_ptr
    logical, intent(out) :: f90wrap_force_bounds_in_subcell
    
    this_ptr = transfer(this, this_ptr)
    f90wrap_force_bounds_in_subcell = this_ptr%p%force_bounds_in_subcell
end subroutine f90wrap_remapping_cs__get__force_bounds_in_subcell

subroutine f90wrap_remapping_cs__set__force_bounds_in_subcell(this, &
    f90wrap_force_bounds_in_subcell)
    use mom_remapping, only: remapping_cs
    implicit none
    type remapping_cs_ptr_type
        type(remapping_cs), pointer :: p => NULL()
    end type remapping_cs_ptr_type
    integer, intent(in)   :: this(2)
    type(remapping_cs_ptr_type) :: this_ptr
    logical, intent(in) :: f90wrap_force_bounds_in_subcell
    
    this_ptr = transfer(this, this_ptr)
    this_ptr%p%force_bounds_in_subcell = f90wrap_force_bounds_in_subcell
end subroutine f90wrap_remapping_cs__set__force_bounds_in_subcell

subroutine f90wrap_remapping_cs_initialise(this)
    use mom_remapping, only: remapping_cs
    implicit none
    
    type remapping_cs_ptr_type
        type(remapping_cs), pointer :: p => NULL()
    end type remapping_cs_ptr_type
    type(remapping_cs_ptr_type) :: this_ptr
    integer, intent(out), dimension(2) :: this
    allocate(this_ptr%p)
    this = transfer(this_ptr, this)
end subroutine f90wrap_remapping_cs_initialise

subroutine f90wrap_remapping_cs_finalise(this)
    use mom_remapping, only: remapping_cs
    implicit none
    
    type remapping_cs_ptr_type
        type(remapping_cs), pointer :: p => NULL()
    end type remapping_cs_ptr_type
    type(remapping_cs_ptr_type) :: this_ptr
    integer, intent(in), dimension(2) :: this
    this_ptr = transfer(this, this_ptr)
    deallocate(this_ptr%p)
end subroutine f90wrap_remapping_cs_finalise

subroutine f90wrap_remapping_core_h(h0, u0, h1, u1, cs, n0, n1, n2)
    use mom_remapping, only: remapping_core_h, remapping_cs
    implicit none
    
    type remapping_cs_ptr_type
        type(remapping_cs), pointer :: p => NULL()
    end type remapping_cs_ptr_type
    double precision, intent(in), dimension(n0) :: h0
    double precision, intent(in), dimension(n1) :: u0
    double precision, intent(in), dimension(n2) :: h1
    double precision, intent(out), dimension(n2) :: u1
    type(remapping_cs_ptr_type) :: cs_ptr
    integer, intent(in), dimension(2) :: cs
    integer :: n0
    !f2py intent(hide), depend(h0) :: n0 = shape(h0,0)
    integer :: n1
    !f2py intent(hide), depend(u0) :: n1 = shape(u0,0)
    integer :: n2
    !f2py intent(hide), depend(h1) :: n2 = shape(h1,0)
    cs_ptr = transfer(cs, cs_ptr)
    call remapping_core_h(n0=n0, h0=h0, u0=u0, n1=n2, h1=h1, u1=u1, CS=cs_ptr%p)
end subroutine f90wrap_remapping_core_h

subroutine f90wrap_mom_remapping__get__remapping_pcm(f90wrap_remapping_pcm)
    use mom_remapping, only: mom_remapping_remapping_pcm => remapping_pcm
    implicit none
    integer, intent(out) :: f90wrap_remapping_pcm
    
    f90wrap_remapping_pcm = mom_remapping_remapping_pcm
end subroutine f90wrap_mom_remapping__get__remapping_pcm

subroutine f90wrap_mom_remapping__get__remapping_plm(f90wrap_remapping_plm)
    use mom_remapping, only: mom_remapping_remapping_plm => remapping_plm
    implicit none
    integer, intent(out) :: f90wrap_remapping_plm
    
    f90wrap_remapping_plm = mom_remapping_remapping_plm
end subroutine f90wrap_mom_remapping__get__remapping_plm

subroutine &
    f90wrap_mom_remapping__get__remapping_ppm_h4(f90wrap_remapping_ppm_h4)
    use mom_remapping, only: mom_remapping_remapping_ppm_h4 => remapping_ppm_h4
    implicit none
    integer, intent(out) :: f90wrap_remapping_ppm_h4
    
    f90wrap_remapping_ppm_h4 = mom_remapping_remapping_ppm_h4
end subroutine f90wrap_mom_remapping__get__remapping_ppm_h4

subroutine &
    f90wrap_mom_remapping__get__remapping_ppm_ih4(f90wrap_remapping_ppm_ih4)
    use mom_remapping, only: mom_remapping_remapping_ppm_ih4 => remapping_ppm_ih4
    implicit none
    integer, intent(out) :: f90wrap_remapping_ppm_ih4
    
    f90wrap_remapping_ppm_ih4 = mom_remapping_remapping_ppm_ih4
end subroutine f90wrap_mom_remapping__get__remapping_ppm_ih4

subroutine &
    f90wrap_mom_remapping__get__remapping_pqm_ih4ih3(f90wrap_remapping_pqm_ih4ih3)
    use mom_remapping, only: mom_remapping_remapping_pqm_ih4ih3 => &
        remapping_pqm_ih4ih3
    implicit none
    integer, intent(out) :: f90wrap_remapping_pqm_ih4ih3
    
    f90wrap_remapping_pqm_ih4ih3 = mom_remapping_remapping_pqm_ih4ih3
end subroutine f90wrap_mom_remapping__get__remapping_pqm_ih4ih3

subroutine &
    f90wrap_mom_remapping__get__remapping_pqm_ih6ih5(f90wrap_remapping_pqm_ih6ih5)
    use mom_remapping, only: mom_remapping_remapping_pqm_ih6ih5 => &
        remapping_pqm_ih6ih5
    implicit none
    integer, intent(out) :: f90wrap_remapping_pqm_ih6ih5
    
    f90wrap_remapping_pqm_ih6ih5 = mom_remapping_remapping_pqm_ih6ih5
end subroutine f90wrap_mom_remapping__get__remapping_pqm_ih6ih5

subroutine f90wrap_mom_remapping__get__integration_pcm(f90wrap_integration_pcm)
    use mom_remapping, only: mom_remapping_integration_pcm => integration_pcm
    implicit none
    integer, intent(out) :: f90wrap_integration_pcm
    
    f90wrap_integration_pcm = mom_remapping_integration_pcm
end subroutine f90wrap_mom_remapping__get__integration_pcm

subroutine f90wrap_mom_remapping__get__integration_plm(f90wrap_integration_plm)
    use mom_remapping, only: mom_remapping_integration_plm => integration_plm
    implicit none
    integer, intent(out) :: f90wrap_integration_plm
    
    f90wrap_integration_plm = mom_remapping_integration_plm
end subroutine f90wrap_mom_remapping__get__integration_plm

subroutine f90wrap_mom_remapping__get__integration_ppm(f90wrap_integration_ppm)
    use mom_remapping, only: mom_remapping_integration_ppm => integration_ppm
    implicit none
    integer, intent(out) :: f90wrap_integration_ppm
    
    f90wrap_integration_ppm = mom_remapping_integration_ppm
end subroutine f90wrap_mom_remapping__get__integration_ppm

subroutine f90wrap_mom_remapping__get__integration_pqm(f90wrap_integration_pqm)
    use mom_remapping, only: mom_remapping_integration_pqm => integration_pqm
    implicit none
    integer, intent(out) :: f90wrap_integration_pqm
    
    f90wrap_integration_pqm = mom_remapping_integration_pqm
end subroutine f90wrap_mom_remapping__get__integration_pqm

subroutine f90wrap_mom_remapping__get__mod(f90wrap_mod)
    use mom_remapping, only: mom_remapping_mod => mod
    implicit none
    character(40), intent(out) :: f90wrap_mod
    
    f90wrap_mod = mom_remapping_mod
end subroutine f90wrap_mom_remapping__get__mod

subroutine f90wrap_mom_remapping__set__mod(f90wrap_mod)
    use mom_remapping, only: mom_remapping_mod => mod
    implicit none
    character(40), intent(in) :: f90wrap_mod
    
    mom_remapping_mod = f90wrap_mod
end subroutine f90wrap_mom_remapping__set__mod

subroutine &
    f90wrap_mom_remapping__get__remappingschemesdoc(f90wrap_remappingschemesdoc)
    use mom_remapping, only: mom_remapping_remappingschemesdoc => &
        remappingschemesdoc
    implicit none
    character(256), intent(out) :: f90wrap_remappingschemesdoc
    
    f90wrap_remappingschemesdoc = mom_remapping_remappingschemesdoc
end subroutine f90wrap_mom_remapping__get__remappingschemesdoc

subroutine &
    f90wrap_mom_remapping__set__remappingschemesdoc(f90wrap_remappingschemesdoc)
    use mom_remapping, only: mom_remapping_remappingschemesdoc => &
        remappingschemesdoc
    implicit none
    character(256), intent(in) :: f90wrap_remappingschemesdoc
    
    mom_remapping_remappingschemesdoc = f90wrap_remappingschemesdoc
end subroutine f90wrap_mom_remapping__set__remappingschemesdoc

subroutine &
    f90wrap_mom_remapping__get__remappingdefaultscheme(f90wrap_remappingdefaultscheme)
    use mom_remapping, only: mom_remapping_remappingdefaultscheme => &
        remappingdefaultscheme
    implicit none
    character(3), intent(out) :: f90wrap_remappingdefaultscheme
    
    f90wrap_remappingdefaultscheme = mom_remapping_remappingdefaultscheme
end subroutine f90wrap_mom_remapping__get__remappingdefaultscheme

subroutine &
    f90wrap_mom_remapping__set__remappingdefaultscheme(f90wrap_remappingdefaultscheme)
    use mom_remapping, only: mom_remapping_remappingdefaultscheme => &
        remappingdefaultscheme
    implicit none
    character(3), intent(in) :: f90wrap_remappingdefaultscheme
    
    mom_remapping_remappingdefaultscheme = f90wrap_remappingdefaultscheme
end subroutine f90wrap_mom_remapping__set__remappingdefaultscheme

subroutine f90wrap_mom_remapping__get__h_neglect(f90wrap_h_neglect)
    use mom_remapping, only: mom_remapping_h_neglect => h_neglect
    implicit none
    real, intent(out) :: f90wrap_h_neglect
    
    f90wrap_h_neglect = mom_remapping_h_neglect
end subroutine f90wrap_mom_remapping__get__h_neglect

subroutine f90wrap_mom_remapping__get__old_algorithm(f90wrap_old_algorithm)
    use mom_remapping, only: mom_remapping_old_algorithm => old_algorithm
    implicit none
    logical, intent(out) :: f90wrap_old_algorithm
    
    f90wrap_old_algorithm = mom_remapping_old_algorithm
end subroutine f90wrap_mom_remapping__get__old_algorithm

! End of module mom_remapping defined in file mom6/src/ALE/MOM_remapping.F90

