module MOM_error_handler

  implicit none ; private

  public MOM_error, FATAL

  integer, parameter :: FATAL = 0

  contains

  subroutine MOM_error(level, message, all_print)
    integer,           intent(in) :: level
    character(len=*),  intent(in) :: message
    logical, optional, intent(in) :: all_print
  end subroutine MOM_error

end module MOM_error_handler
