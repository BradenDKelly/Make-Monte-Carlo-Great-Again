import numpy as np
import json, sys

import MMMCGA_auxilary
from MMMCGA_blocks import blk_begin, blk_end, blk_add, run_begin, run_end
from MMMCGA_auxilary import introduction, conclusion, potential, potential_1, PotentialType
from MMMCGA_auxilary import metropolis, random_translate_vector, PrintPDB, InitCubicGrid

from KolafaNezbeda import ULJ, PressureLJ, FreeEnergyLJ_res


def calc_variables():
    """Calculates all variables of interest.
    They are collected and returned as a list, for use in the main program.
    """

    # In this example we simulate using the cut (but not shifted) potential
    # The values of < p_c >, < e_c > and density should be consistent (for this potential)
    # For comparison, long-range corrections are also applied to give
    # estimates of < e_f > and < p_f > for the full (uncut) potential
    # The value of the cut-and-shifted potential is not used, in this example

    import math
    from MMMCGA_auxilary import potential_lrc, pressure_lrc, pressure_delta, VariableType
    # from averages_module import msd, VariableType
    # from lrc_module import potential_lrc, pressure_lrc, pressure_delta
    # from mc_lj_module import force_sq

    # Preliminary calculations (n,r,total are taken from the calling program)
    vol = box ** 3  # Volume
    rho = nAtoms / vol  # Density
    # fsq = force_sq(box, r_cut, r)  # Total squared force

    # Variables of interest, of class VariableType, containing three attributes:
    #   .val: the instantaneous value
    #   .nam: used for headings
    #   .method: indicating averaging method
    # If not set below, .method adopts its default value of avg
    # The .nam and some other attributes need only be defined once, at the start of the program,
    # but for clarity and readability we assign all the values together below

    # Move acceptance ratio
    m_r = VariableType(nam='Move ratio', val=m_ratio, instant=False)

    # Internal energy per atom for simulated, cut, potential
    # Ideal gas contribution plus cut (but not shifted) PE divided by N
    e_c = VariableType(nam='E/N cut', val=1.5 * temperature + total.pot / nAtoms)

    # Internal energy per atom for full potential with LRC
    # LRC plus ideal gas contribution plus cut (but not shifted) PE divided by N
    e_f = VariableType(nam='E/N full', val=potential_lrc(rho, r_cut) + 1.5 * temperature + total.pot / nAtoms)

    # Residual energy per atom for full potential with LRC
    # LRC plus cut (but not shifted) PE divided by N
    e_r = VariableType(nam='E/N Residual', val=potential_lrc(rho, r_cut) + total.pot / nAtoms)

    # KolafaNezbeda EOS Residual energy per atom for full potential with LRC
    # LRC plus cut (but not shifted) PE divided by N
    eos_r = VariableType(nam='E/N EOS ', val=ULJ(temperature, rho))

    # KolafaNezbeda EOS LJ Pressure
    eos_p = VariableType(nam='P EOS ', val=PressureLJ(temperature, rho))

    # KolafaNezbeda EOS LJ Chemical Potential
    eos_mu = VariableType(nam='Chem Pot EOS ', val=FreeEnergyLJ_res(temperature, rho))

    # Pressure for simulated, cut, potential
    # Delta correction plus ideal gas contribution plus total virial divided by V
    p_c = VariableType(nam='P cut', val=pressure_delta(rho, r_cut) + rho * temperature + total.vir / vol)

    # Pressure for full potential with LRC
    # LRC plus ideal gas contribution plus total virial divided by V
    p_f = VariableType(nam='P full', val=pressure_lrc(rho, r_cut) + rho * temperature + total.vir / vol)

    # Configurational temperature
    # Total squared force divided by total Laplacian
    # t_c = VariableType(nam='T config', val=fsq / total.lap)

    # Heat capacity (full)
    # MSD potential energy divided by temperature and sqrt(N) to make result intensive; LRC does not contribute
    # We add ideal gas contribution, 1.5, afterwards
    c_f = VariableType(nam='Cv/N full', val=total.pot / (temperature * math.sqrt(nAtoms)),
                       method=msd, add=1.5, instant=False)

    # Collect together into a list for averaging
    return [m_r, p_c, p_f, e_c, e_f, e_r, c_f, eos_r, eos_p, eos_mu]  # , c_f,t_c,


np.random.seed(111)

"""
temperature = 1.0 #0.8772  # 1.2996
rho = 0.75
nAtoms = 256
epsilon = 1.0
sigma = 1.0

r_cut = 2.5  # box / 2
nSteps = 100
nblock = 100
#dr_max = box / 75
outputInterval = 100
initialConfiguration = 'crystal'  # place atoms in a crystal structure
"""
# Read parameters in JSON format
try:
    nml = json.load(open("input.inp"))
except json.JSONDecodeError:
    print('Exiting on Invalid JSON format')
    sys.exit()

# Set default values, check keys and typecheck values
defaults = {"nblock": 10, "nstep": 1000, "temperature": 1.0, "r_cut": 2.5, "dr_max": 0.15, \
            "natoms": 256, "initConfig": "crystal"}

for key, val in nml.items():
    if key in defaults:
        assert type(val) == type(defaults[key]), key + " has the wrong type"
    else:
        print('Warning', key, 'not in ', list(defaults.keys()))

# Set parameters to input values or defaults
nblock = nml["nblock"] if "nblock" in nml else defaults["nblock"]
nSteps = nml["nstep"] if "nstep" in nml else defaults["nstep"]
temperature = nml["temperature"] if "temperature" in nml else defaults["temperature"]
r_cut = nml["r_cut"] if "r_cut" in nml else defaults["r_cut"]
dr_max = nml["dr_max"] if "dr_max" in nml else defaults["dr_max"]
nAtoms = nml["natoms"] if "natoms" in nml else defaults["natoms"]
epsilon = nml["epsilon"] if "epsilon" in nml else print("no epsilon in input file")
sigma = nml["sigma"] if "sigma" in nml else print("no sigma in input file")
rho = nml["rho"] if "rho" in nml else print("no rho in input file")
outputInterval = nml["outputInterval"] if "outputInterval" in nml else print("no outputInterval in input file")
initialConfiguration = nml["initConf"] if "initConf" in nml else defaults["initConf"]

box = (nAtoms / rho) ** (1 / 3)

print(nSteps, box, dr_max, r_cut)

if "crystal" in initialConfiguration.lower():
    r = InitCubicGrid(nAtoms, rho)
else:
    r = np.random.rand(nAtoms, 3) * box
    print("WARNING: overlap in energy routines do not handle overlap properly. Random init config not advised")
r = r - np.rint(r / box) * box  # Periodic boundaries

avg = 0
msd = 1
cke = 2
PrintPDB(r, 1, "test")
# Initial energy and overlap check
total = potential(box, r_cut, r)
print("PRINTING: ", total.pot, total.vir)
# assert not total.ovr, 'Overlap in initial configuration'

# Initialize arrays for averaging and write column headings
m_ratio = 0.0
m_accept = 0
m_trial = 0

introduction()
n_avg = run_begin(calc_variables())

for blk in range(1, nblock + 1):  # Loop over blocks

    blk_begin(n_avg)

    for stp in range(nSteps):  # Loop over steps

        moves = 0

        for i in range(nAtoms):  # Loop over atoms
            m_trial += 1
            rj = np.delete(r, i, 0)  # Array of all the other atoms
            partial_old = potential_1(r[i, :], box, r_cut, rj)  # Old atom potential, virial etc
            if stp > 100: assert not partial_old.ovr, 'Overlap in current configuration'

            ri = random_translate_vector(dr_max, r[i, :])  # Trial move to new position (in box=1 units)
            ri = ri - np.rint(ri / box) * box  # Periodic boundary correction

            partial_new = potential_1(ri, box, r_cut, rj)  # New atom potential, virial etc
            if not partial_new.ovr:  # Test for non-overlapping configuration
                delta = partial_new.pot - partial_old.pot  # Use cut (but not shifted) potential
                delta = delta / temperature
                if metropolis(delta):  # Accept Metropolis test
                    total = total + partial_new - partial_old  # Update total values
                    r[i, :] = ri  # Update position
                    moves = moves + 1  # Increment move counter
                    m_accept += 1

        m_ratio = m_accept / m_trial  # moves / nAtoms

        blk_add(calc_variables(), n_avg)
    # print(min(r[0]),min(r[1]),min(r[2]),max(r[0]),max(r[1]),max(r[2]))
    PrintPDB(r, stp, "test")
    blk_end(blk)  # Output block averages
    # sav_tag = str(blk).zfill(3) if blk<1000 else 'sav'    # Number configuration by block
    # write_cnf_atoms ( cnf_prefix+sav_tag, n, box, r*box ) # Save configuration

run_end(calc_variables())

total = potential(box, r_cut, r)  # Double check book-keeping
# assert not total.ovr, 'Overlap in final configuration'

# ( cnf_prefix+out_tag, n, box, r*box ) # Save configuration
conclusion()
