#!/usr/bin/env python3
# mc_nvt_poly_lj.py

# ------------------------------------------------------------------------------------------------#
# This software was written in 2016/17                                                           #
# by Michael P. Allen <m.p.allen@warwick.ac.uk>/<m.p.allen@bristol.ac.uk>                        #
# and Dominic J. Tildesley <d.tildesley7@gmail.com> ("the authors"),                             #
# to accompany the book "Computer Simulation of Liquids", second edition, 2017 ("the text"),     #
# published by Oxford University Press ("the publishers").                                       #
#                                                                                                #
# LICENCE                                                                                        #
# Creative Commons CC0 Public Domain Dedication.                                                 #
# To the extent possible under law, the authors have dedicated all copyright and related         #
# and neighboring rights to this software to the PUBLIC domain worldwide.                        #
# This software is distributed without any warranty.                                             #
# You should have received a copy of the CC0 Public Domain Dedication along with this software.  #
# If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.                               #
#                                                                                                #
# DISCLAIMER                                                                                     #
# The authors and publishers make no warranties about the software, and disclaim liability       #
# for all uses of the software, to the fullest extent permitted by applicable law.               #
# The authors and publishers do not recommend use of this software for any purpose.              #
# It is made freely available, solely to clarify points made in the text. When using or citing   #
# the software, you should not imply endorsement by the authors or publishers.                   #
# ------------------------------------------------------------------------------------------------#

"""Monte Carlo, NVT ensemble, polyatomic molecule."""


def calc_variables():
    """Calculates all variables of interest.
    They are collected and returned as a list, for use in the main program.
    """

    # In this example we simulate using the shifted-force potential only
    # The values of < p_sf >, < e_sf > and density should be consistent (for this potential)
    # There are no long-range or delta corrections

    from averages_module import VariableType

    # Preliminary calculations
    vol = box ** 3  # Volume
    rho = n / vol  # Density

    # Variables of interest, of class VariableType, containing three attributes:
    #   .val: the instantaneous value
    #   .nam: used for headings
    #   .method: indicating averaging method
    # If not set below, .method adopts its default value of avg
    # The .nam and some other attributes need only be defined once, at the start of the program,
    # but for clarity and readability we assign all the values together below

    # Move acceptance ratio
    m_r = VariableType(nam='Move ratio', val=m_ratio, instant=False)

    # Internal energy per molecule (shifted-force potential)
    # Ideal gas contribution (assuming nonlinear molecules) plus total PE divided by N
    e_sf = VariableType(nam='E/N shifted force', val=3.0 * temperature + total.pot / n)

    # Pressure (shifted-force potential)
    # Ideal gas contribution plus total virial divided by V
    p_sf = VariableType(nam='P shifted force', val=rho * temperature + total.vir / vol)

    # Collect together into a list for averaging
    return [m_r, e_sf, p_sf]


# Takes in a configuration of polyatomic molecules (positions and quaternions)
# Cubic periodic boundary conditions
# Conducts Monte Carlo at the given temperature
# Uses no special neighbour lists

# Reads several variables and options from standard input using JSON format
# Leave input empty "{}" to accept supplied defaults

# Positions r are divided by box length after reading in
# However, input configuration, output configuration, most calculations, and all results
# are given in simulation units defined by the model
# For example, for Lennard-Jones, sigma = 1, epsilon = 1

# Despite the program name, there is nothing here specific to Lennard-Jones
# The model (including the cutoff distance) is defined in mc_poly_lj_module

import json
import sys
import numpy as np
import time
import math
from config_io_module import read_cnf_mols, write_cnf_mols, PrintPDB
from averages_module import run_begin, run_end, blk_begin, blk_end, blk_add
from maths_module import random_translate_vector, metropolis, random_rotate_quaternion, q_to_a
from mc_poly_lj_module import introduction, conclusion, potential, potential_1, PotentialType, na, db, potential_2
from initialize import fcc_positions_2, ran_positions

import energy_f2py # f2py_energy #
#from energy_f2py import potential_3
#import f2py_energy
#print("TESTING")
#print(f2py_energy.__doc__)

"""
In terminal run

python -m numpy.f2py -m energy_f2py energy_f2py.f90 -h energy_f2py.pyf

then

python -m numpy.f2py -c energy_f2py.pyf energy_f2py.f90

this creates the f2py module
"""

cnf_prefix = 'cnf.'
inp_tag = 'inp'
out_tag = 'out'
sav_tag = 'sav'

print('mc_nvt_poly_lj')
print('Monte Carlo, constant-NVT ensemble, polyatomic molecule')
print('Simulation uses cut-and-shifted potential')

# Read parameters in JSON format
#try:
#    nml = json.load(sys.stdin)
"""
try:
    nml = json.load(open("input.inp"))
except json.JSONDecodeError:
    print('Exiting on Invalid JSON format')
    sys.exit()
"""
nml={"nblock": 10, "nstep": 10000, "temperature": 0.6, "dr_max": 0.05, "de_max": 0.05}
# Set default values, check keys and typecheck values
defaults = {"nblock": 10, "nstep": 1000, "temperature": 1.0, "dr_max": 0.05, "de_max": 0.05}
for key, val in nml.items():
    if key in defaults:
        assert type(val) == type(defaults[key]), key + " has the wrong type"
    else:
        print('Warning', key, 'not in ', list(defaults.keys()))

# Set parameters to input values or defaults
nblock = nml["nblock"] if "nblock" in nml else defaults["nblock"]
nstep = nml["nstep"] if "nstep" in nml else defaults["nstep"]
temperature = nml["temperature"] if "temperature" in nml else defaults["temperature"]
dr_max = nml["dr_max"] if "dr_max" in nml else defaults["dr_max"]
de_max = nml["de_max"] if "de_max" in nml else defaults["de_max"]

introduction()
np.random.seed(111)

# Write out parameters
print("{:40}{:15d}  ".format('Number of blocks', nblock))
print("{:40}{:15d}  ".format('Number of steps per block', nstep))
print("{:40}{:15.6f}".format('Specified temperature', temperature))
print("{:40}{:15.6f}".format('Maximum r displacement', dr_max))
print("{:40}{:15.6f}".format('Maximum e displacement', de_max))

# Read in initial configuration
manual = False
if manual:
    n = 4**3*4
    rho = 0.32655
    box = (n/rho)**(1/3) #10.0
    r, e = ran_positions(n, box, 0.0, False, True)
else:
    n, box, r, e = read_cnf_mols(cnf_prefix + inp_tag, quaternions=True)
    print("using config from Tildesley")
#r,e,= fcc_positions_2(n, box)
#print("number molecules...: ", len(r))
#print("shape: ", np.shape(r))

# tracks which atoms are in which molecule



print("{:40}{:15d}  ".format('Number of particles', n))
print("{:40}{:15.6f}".format('Box length', box))
print("{:40}{:15.6f}".format('Density', n / box ** 3))
r = r / box  # Convert positions to box units
r = r - np.rint(r)  # Periodic boundaries

atom_mol=np.zeros((len(r),2),dtype=np.int_)
atom_mol[:,0]=np.arange(0,len(r)*3,3)
atom_mol[:,1]=np.arange(2,len(r)*3+2,3)

# Calculate all bond vectors
d = np.empty((n, na, 3), dtype=np.float_)
a = []
for i, ei in enumerate(e):
    ai = q_to_a(ei)  # Rotation matrix for i
    d[i, :, :] = np.dot(db, ai)  # NB: equivalent to ai_T*db, ai_T=transpose of ai
    for j in range(3):
       a.append(r[i,:]*box + np.dot(db[j,:], ai))
a = np.asarray(a)
#print("atom list: ", np.shape(a))
PrintPDB(a,"001","test")
PrintPDB(r,"001","testing")
atom_com = np.asarray([np.mean(a[0:3,0]), np.mean(a[0:3,1]), np.mean(a[0:3,2])])
print("COM atoms: ", atom_com)
print("molecule: ", r[0]*box)
if max(abs(atom_com - r[0])) > 0.0001:
    print("issues with atom and molecules center of masses - line 177")
# Initial energy and overlap check
total = potential(box, r, d)
assert not total.ovr, 'Overlap in initial configuration'

# Initialize arrays for averaging and write column headings
quat = False #True
m_ratio = 0.0
diameter = 2.0 * np.sqrt ( np.max ( np.sum(db**2,axis=1) ) ) # Molecular diameter
r_cut    = 2.612
rm_cut_box = (r_cut + diameter) / box  # Molecular cutoff in box=1 units
rm_cut_box_sq = rm_cut_box ** 2
run_begin(calc_variables())

f2py = False #True

for blk in range(1, nblock + 1):  # Loop over blocks

    blk_begin()

    for stp in range(nstep):  # Loop over steps

        moves = 0

        for i in range(n):  # Loop over atoms

            if quat:
                rj = np.delete(r, i, 0)  # Array of all the other molecules
                dj = np.delete(d, i, 0)  # Array of all the other molecules
                partial_old = potential_1(r[i, :], d[i, :, :], box, rj, dj)  # Old potential, virial etc
                #print(i, partial_old.pot)
            elif f2py:
                rj = np.delete(r, i, 0)  # Array of all the other molecules
                dj = np.delete(d, i, 0)  # Array of all the other molecules
                na = 3
                nm, fill = np.shape(rj)
                energy, virial, overlap = f2py_energy.energy_f2py.potential_3(r[i, :], d[i, :, :], i, box, rj, dj, na, nm, rm_cut_box_sq)
                partial_old = PotentialType(pot=energy, vir=virial, ovr=overlap)
            else:

                a1, a2, rj = atom_mol[i,0], atom_mol[i,1]+1, np.delete(r, i, 0)
                aj = np.delete(a, np.arange(a1,a2), 0)
                partial_old = potential_2(a[a1:(a2), :], r[i,:], box, aj, rj)
            assert not partial_old.ovr, 'Overlap in current configuration'

            ri = random_translate_vector(dr_max / box, r[i, :])  # Trial move to new position (in box=1 units)
            ri = ri - np.rint(ri)  # Periodic boundary correction
            ei = random_rotate_quaternion(de_max, e[i, :])  # Trial rotation
            ai = q_to_a(ei)  # Rotation matrix for i
            #di = np.dot(db, ai)  # NB: equivalent to ai_T*db, ai_T=transpose of ai

            if quat:
                partial_new = potential_1(ri, di, box, rj, dj)  # New atom potential, virial etc
            else:
                ad = []
                for j in range(3):
                    ad.append(ri * box + np.dot(db[j, :], ai))
                ad = np.asarray(ad)
                partial_new = potential_2(ad, ri, box, aj, rj)

            if not partial_new.ovr:  # Test for non-overlapping configuration
                delta = partial_new.pot - partial_old.pot  # Use cut (but not shifted) potential
                delta = delta / temperature
                if metropolis(delta):  # Accept Metropolis test
                    total = total + partial_new - partial_old  # Update total values
                    r[i, :] = ri  # Update position
                    e[i, :] = ei  # Update quaternion
                    #d[i, :, :] = di  # Update bond vectors
                    if not quat:
                        a[a1:a2, :] = ad
                    moves = moves + 1  # Increment move counter
        #if NPT:
        #    partial_old = total
        #    partial_new = VolumeChange(r,e,a,box,dV_max,)
        m_ratio = moves / n

        blk_add(calc_variables())

    blk_end(blk)  # Output block averages
    sav_tag = str(blk).zfill(3) if blk < 1000 else 'sav'  # Number configuration by block
    write_cnf_mols(cnf_prefix + sav_tag, n, box, r * box, e)  # Save configuration

run_end(calc_variables())
PrintPDB(a,"003","final")
total = potential(box, r, d)  # Double check book-keeping
assert not total.ovr, 'Overlap in final configuration'

write_cnf_mols(cnf_prefix + out_tag, n, box, r * box, e)  # Save configuration

conclusion()

