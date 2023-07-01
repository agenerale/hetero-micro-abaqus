import numpy as np
import os
import sys
import argparse
import shutil
from abaqus import *
from abaqusConstants import *
from caeModules import *
from odbAccess import *
from driverUtils import executeOnCaeStartup
from abaqus_helpers import *
#-------------------------------------------------------
'''
Runs Mechanical and Thermal Analyses through Abaqus
on 2D heterogeneous microstructures and calculates
effective elastic properties and thermal conductivity.

Runs through all microstructures in order of:
1. Thermal cases
2. Mechanical cases
'''
#-------------------------------------------------------
# Usage: abaqus cae noGUI=abaqus_array.py -- 0
microAll = np.load('micros_hetero.npy')
inc = 1000
micro_init = int(sys.argv[-1])*inc
micro_end = micro_init + inc

if micro_end > 9990:
    micro_end = 9990
    inc = micro_end - micro_init

# Create empty array to record results
output = np.zeros((inc, 3))
    
k = 0
for i in range(micro_init,micro_end,1):
    # Select microstructure to run from array
    print >> sys.__stdout__, 'Micro #:' + str(i)
    hetero = microAll[k,...,None]
    hetero = np.concatenate([hetero, 1-hetero], axis=-1)

    # Run Abaqus
    jobName = 'micro_pbc_' + str(i)
        
    #PBC_Homogenization_2D_Thermal(
    #    hetero_micro=hetero,
    #    job_location=jobName,
    #    num_cpu_job = 4,
    #    thermal_properties=((247), (400))
    #)
    
    PBC_Homogenization_2D_Thermal(
        hetero_micro=hetero,
        job_location=jobName,
        num_cpu_job = 4,
        boundary_conditions = (
            (1.0, 0.0),
            (0.0, 1.0),
        ),
        thermal_properties=((1), (1000))
        #thermal_properties=((247), (400))
    )    
    
    print >> sys.__stdout__, 'Micro #:' + str(i) + ' complete analysis'
    
    # Post-process for results form .odb file
    kx, ky = Thermal_Post(jobName)
    output[k,:] = np.array([i, kx, ky])
    
    # Remove associated abaqus files
    os.remove(jobName + '.odb')
    os.remove(jobName + '.dat')
    os.remove(jobName + '.inp')
    os.remove(jobName + '.log')
    os.remove(jobName + '.com')
    os.remove(jobName + '.msg')
    os.remove(jobName + '.prt')
    os.remove(jobName + '.sta')
    print >> sys.__stdout__, 'Micro #:' + str(i) + ' complete post-processing'
    
    k += 1

# Change back to top level directory and write results
outfile = str(micro_init) + '_thermal.csv'
np.savetxt(outfile,output,delimiter=',')

#-------------------------------------------------------
# Create empty array to record results
output = np.zeros((inc, 6))
    
k = 0
for i in range(micro_init,micro_end,1):
    # Select microstructure to run from array
    print >> sys.__stdout__, 'Micro #:' + str(i)
    hetero = microAll[k,...,None]
    hetero = np.concatenate([hetero, 1-hetero], axis=-1)

    # Run Abaqus
    jobName = 'micro_pbc_' + str(i)
        
    PBC_Homogenization_2D(
        hetero_micro=hetero,
        job_location=jobName,
        num_cpu_job = 4,
        #scratch_dir = dir,
        mech_properties=((10.E3, 0.3), (1000.E3, 0.3))
        #mech_properties=((70.E3, 0.33), (169.E3, 0.28))
    )
    
    print >> sys.__stdout__, 'Micro #:' + str(i) + ' complete analysis'
    
    # Post-process for results form .odb file
    E0_x, v0_xy, E0_y, v0_yx, G0_xy = Plane_Stress_Mechanical_Post(jobName)
    output[k,:] = np.array([i, E0_x, v0_xy, E0_y, v0_yx, G0_xy])
    
    # Remove associated abaqus files
    os.remove(jobName + '.odb')
    os.remove(jobName + '.dat')
    os.remove(jobName + '.inp')
    os.remove(jobName + '.log')
    os.remove(jobName + '.com')
    os.remove(jobName + '.msg')
    os.remove(jobName + '.prt')
    os.remove(jobName + '.sta')
    print >> sys.__stdout__, 'Micro #:' + str(i) + ' complete post-processing'
    
    k += 1

# Change back to top level directory and write results
outfile = str(micro_init) + '_mech.csv'
np.savetxt(outfile,output,delimiter=',')
    
    
    


