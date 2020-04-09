module energy_f2py
integer kkk
contains
  SUBROUTINE potential_3(  ri, di, i, box, r, d, na, nm, rm_cut_box_sq, energy, virial, overlap)
    IMPLICIT NONE
    !TYPE(potential_type)                 :: partial ! Returns a composite of pot, vir etc for given molecule
    INTEGER,                  INTENT(in) :: na, nm
    REAL,    DIMENSION(3),    INTENT(in) :: ri      ! Coordinates of molecule of interest
    REAL,    DIMENSION(3,na), INTENT(in) :: di      ! Bond vectors of molecule of interest
    REAL,    DIMENSION(3,nm), INTENT(in) :: r       ! coordinates and bond vectors of other molecules
    REAL,    DIMENSION(0:3,na,nm), INTENT(in) :: d ! bond vectors
    INTEGER,                  INTENT(in) :: i       ! Index of molecule of interest
    REAL,                     INTENT(in) :: box, rm_cut_box_sq    ! Simulation box length

    REAL,                     INTENT(out) :: energy, virial
    LOGICAL,                  INTENT(out) :: overlap

    ! It is assumed that r has been divided by box
    ! Results are in LJ units where sigma = 1, epsilon = 1
    ! Note that this is the force-shifted LJ potential with a linear smoothing term
    ! S Mossa, E La Nave, HE Stanley, C Donati, F Sciortino, P Tartaglia, Phys Rev E, 65, 041205 (2002)

    INTEGER               :: j, a, b
    REAL                  :: sr2, sr6, sr12, rij_sq, rab_sq, virab, rmag, vir, pot, r_cut_sq
    REAL, DIMENSION(3)    :: rij, rab, fab
    REAL, PARAMETER       :: sr2_ovr = 1.77 ! overlap threshold (pot > 100)
    LOGICAL               :: ovr = .FALSE.

  REAL, PARAMETER :: r_cut   = 2.612 ! in sigma=1 units, where r_cut = 1.2616 nm, sigma = 0.483 nm
  REAL, PARAMETER :: sr_cut  = 1.0/r_cut, sr_cut6 = sr_cut**6, sr_cut12 = sr_cut6**2
  REAL, PARAMETER :: lambda1 = 4.0*(7.0*sr_cut6-13.0*sr_cut12)
  REAL, PARAMETER :: lambda2 = -24.0*(sr_cut6-2.0*sr_cut12)*sr_cut

r_cut_sq = r_cut **2

ENERGY = 0.0
vir = 0.0
pot = 0.0
virial = 0.0
overlap = .FALSE.


DO j = 1, nm ! Loop over selected range of partner molecules

   IF ( i == j ) CYCLE ! Skip self

   rij(:) = ri(:) - r(j,:)            ! Centre-centre separation vector
   rij(:) = rij(:) - ANINT ( rij(:) ) ! Periodic boundaries in box=1 units
   rij_sq = SUM ( rij**2 )            ! Squared centre-centre separation in box=1 units

   IF ( rij_sq < rm_cut_box_sq ) THEN ! Test within molecular cutoff

      rij = rij * box ! Now in sigma = 1 units

      ! Double loop over atoms on both molecules
      DO a = 1, na
         DO b = 1, na

            rab    = rij + di(a,:) - d(j,b,:) ! Atom-atom vector, sigma=1 units
            rab_sq = SUM ( rab**2 )           ! Squared atom-atom separation, sigma=1 units

            IF ( rab_sq < r_cut_sq ) THEN ! Test within potential cutoff

               sr2      = 1.0 / rab_sq  ! (sigma/rab)**2
               ovr = sr2 > sr2_ovr ! Overlap if too close

               IF ( ovr ) THEN
                  overlap = .TRUE. ! Overlap detected
                  RETURN               ! Return immediately
               END IF

               rmag     = SQRT(rab_sq)
               sr6      = sr2**3
               sr12     = sr6**2
               pot      = 4.0*(sr12-sr6) + lambda1 + lambda2*rmag ! LJ atom-atom pair potential (force-shifted)
               virab    = 24.0*(2.0*sr12-sr6 ) - lambda2*rmag     ! LJ atom-atom pair virial
               fab      = rab * virab * sr2                       ! LJ atom-atom pair force
               vir      = DOT_PRODUCT ( rij, fab )                ! Contribution to molecular virial

               energy = energy + pot
               virial = virial + vir

            END IF ! End test within potential cutoff

         END DO
      END DO
      ! End double loop over atoms on both molecules

   END IF ! End test within molecular cutoff

END DO ! End loop over selected range of partner molecules

! Include numerical factors
virial = virial / 3.0 ! Divide virial by 3
overlap = .FALSE.           ! No overlaps detected (redundant, but for clarity)
return
END SUBROUTINE potential_3
end module energy_f2py