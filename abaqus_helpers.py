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

eps = 1e-6 # epsilon for finding the edges of the simulation domain
eps_bc = 1e-8 # epsilon for checking if bc is 0

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------

def get_centroid(element):
    """
    Averages the locations of all of the nodes in an element and
    returns the centroid. 

    Assumes that the element is a square. 
    """
    centroid = []
    for node in element.getNodes():
        centroid.append(
            node.coordinates
        )
    centroid = np.array(centroid).mean(axis=0)
    return centroid

def map_centroid_to_index(centroid, shape):
    """
    Maps the centroid of an element to the equivalent indexes for
    indexing the microstructure array. This assumes that
    the microstructure is a square and that all sides of the domain
    are of length 1.0. 

    :param centroid: (np.ndarray) The centroid of an element. 
    :param shape: (tuple) The shape of the array -- includes just the
                          spatial dimensions 
    """
    index = tuple([int(cent * dim) for cent, dim in zip(centroid, shape)])
    return index

def map_onehot_to_numbered(micro):
    """
    Takes a microstructure with onehot encodings and
    returns a microstructure where the value indicates which
    region is onehot encoded. 
    """
    indicators = (micro * np.arange(micro.shape[-1])).sum(axis=-1).astype(int)
    return indicators

# -------------------------------------------------------------------
# Core Abaqus Functions
# -------------------------------------------------------------------

'''
Things that will eventually change:
(1) 2D or 3D >> implicitly
(2) Where to save everything >> explicitly
(3) What microstructure are we running >> explicitly
(4) plane strain or plane stress >> 
(5) boundary conditions
(6) material parameters -- and material type -- isotropic linear elastic
'''

def PBC_Homogenization_2D(
    hetero_micro,
    job_location,
    num_cpu_job,
    scratch_dir = '',
    plane_stress_flag = True,
    boundary_conditions = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    mech_properties = (
        (1.E3, 0.3),
        (10.E3, 0.3),
    ),
):
    """
    This method initializes and runs a complete job performing linear
    elastic homogenization using periodic boundary conditions. The 
    conditions are adapted from Li's 2000 paper "General Unit Cells
    for Micromechnical Analyses of Unidirectional Composites". Instead
    of Generalized Plain Strain, we only implement plain strain / stress. 

    The method will run analysis on a passed microstructure and
    save an odb file according to the 'job_location' parameter.

    Args:
        hetero_micro (np.ndarray): A numpy array containing the desired
                        microstructure for analysis. The array should be of 
                        the shape: [SPATIAL_DIMENSIONS x STATES]. States should
                        be encoded in a one-hot fashion (i.e., only one state
                        can exist in each voxel at once). 
        job_location (str): The location where the ODB file should be saved
                                and its name. 
        plane_stress_flag (bool, optional): A flag indicating
                        plane stress or plane strain simulations. Defaults to True.
        boundary_conditions (tuple, optional): An array / tuple containing
                    the desired boundary conditions. Each element in the tuple will be
                    applied as a single step in the simulation.
                    Defaults to ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), ).
        mech_properties (tuple, optional): A tuple of the desired
                    mechanical properties of each constituent. 
                    Assumes that each constituent is a isotropic linear
                    elastic material.
                    Defaults to ( (1.E3, 0.3), (10.E3, 0.3), ).
    """
    assert len(boundary_conditions) > 0, 'Must provide at least ' + \
        'one boundary condition.'
    for n, bc in enumerate(boundary_conditions):
        assert len(bc) == 3, 'In 2D, 3 boundary conditions must be given. ' + \
            str(len(bc)) + ' given for the boundary condition at index ' + \
                str(n) + '.'
    assert type(plane_stress_flag) is bool, 'Flags must be boolean.'
    assert len(hetero_micro.shape) == 3, 'Microstructures are expected to ' + \
        'have 2 spatial dimensions and one state dimension (at the ' + \
            'end). Only eigenmicrostructures are supported.'

    # transform microstructure to numbered encoding -- 
    # i.e., (0, 1, 2, ...) for each state       
    hetero_micro = map_onehot_to_numbered(hetero_micro)
    micro_shape = hetero_micro.shape

    assert micro_shape[0] == micro_shape[1], \
        'This implementation only supports squares.'

    # Defining the element type
    if plane_stress_flag:
        # define plane stress elements
        element_types = (CPS4, CPS3)
    else:
        # use plane strain elements
        element_types = (CPE4, CPE3)
        
    # check if there is only 1 active state and define flag
    if len(np.unique(hetero_micro)) == 1:
        localstateflag = np.unique(hetero_micro)
    
    #-----------------------------------------------------
    # Create a model container. 
    myModel = mdb.Model(name='Box')

    #-----------------------------------------------------
    import part
    # Create a sketch for the base feature.

    mySketch = myModel.ConstrainedSketch(name='Box',
        sheetSize=250.)

    mySketch.rectangle(point1=(0,0), point2=(1,1))

    myMicro = myModel.Part(name='Micro', dimensionality=TWO_D_PLANAR,
        type=DEFORMABLE_BODY)

    # tell it that it will be 2D. 
    myMicro.BaseShell(sketch=mySketch)

    myMicro.ReferencePoint(point=(1.0, 1.0, 0.0))

    #-----------------------------------------------------
    import material
    # Create a material and sections
    # IF YOU WANT TO IMPLEMENT A DIFFERENT MATERIAL TYPE
    # (e.g., elastic rigid plastic) YOU SHOULD ONLY NEED 
    # TO CHANGE THE FOLLOWING LINES.

    mats = []
    sections = []

    for n, mech_prop in enumerate(mech_properties):
        # creating the material
        mats.append(
            myModel.Material(name='Mat' + str(n))
        )
        mats[-1].Elastic(table=(mech_prop, ))

        # creating the section
        sections.append(
            myModel.HomogeneousSolidSection(
                name = 'State' + str(n),
                material='Mat' + str(n),
                thickness=1.0,
            )
        )

    #-------------------------------------------------------
    import mesh
    # Mesh the part.
    # e.g., For Plane Strain
    desired_element = mesh.ElemType(elemCode=element_types[0], elemLibrary=STANDARD)
    backup_element = mesh.ElemType(elemCode=element_types[1], elemLibrary=STANDARD)

    faces = myMicro.faces.getSequenceFromMask(mask=('[#1 ]', ),)
    myMicro.setElementType(
        regions=(faces, ), 
        elemTypes=(desired_element, backup_element),
    )

    # Create a mesh with the same number of elements as voxels
    size_value = 1.0 / micro_shape[0]
    myMicro.seedPart(
        size=size_value, 
        deviationFactor=size_value, 
        minSizeFactor=size_value
    )
    myMicro.generateMesh()

    # add DriverNodes
    drivers = []
    drivers = [
        myMicro.Node(coordinates=(0.5, 0.5, 0.0)),
        myMicro.Node(coordinates=(0.5, 0.5, 0.0)),
        myMicro.Node(coordinates=(0.5, 0.5, 0.0)),
    ]

    #--------------------------------------------------------
    # Creating Node and Element Sets

    # Create Node sets for boundary conditions
    all_nodes = myMicro.nodes

    # sides
    top = []
    bottom = []
    left = []
    right = []

    # corners
    topleft = []
    topright = []
    botleft = []
    botright = []

    for node in myMicro.nodes:
        xcoord, ycoord = node.coordinates[:len(micro_shape)]

        if np.abs(xcoord) <= eps:
            # near left edge
            if np.abs(ycoord) <= eps:
                # left bottom corner
                botleft.append(node.label)
            
            elif np.abs(1-ycoord) <= eps:
                # left top corner
                topleft.append(node.label)
            
            else:
                # just on the left edge
                left.append(node.label)

        elif np.abs(1-xcoord) <= eps:
            # near right edge
            if np.abs(ycoord) <= eps:
                # right bottom corner
                botright.append(node.label)
            
            elif np.abs(1-ycoord) <= eps:
                # right top corner
                topright.append(node.label)
            
            else:
                # just on the right edge
                right.append(node.label)
        
        elif np.abs(ycoord) <= eps:
            # bottom -- but no need to check for corners. 
            bottom.append(node.label)
        
        elif np.abs(1-ycoord) <= eps:
            # top -- but no need to check for corners. 
            top.append(node.label)

    # creating the node sets
    # sides
    top = myMicro.SetFromNodeLabels(
        name = 'Top',
        nodeLabels = tuple(top),
        unsorted=True,
    )
    bottom = myMicro.SetFromNodeLabels(
        name = 'Bottom',
        nodeLabels = tuple(bottom),
        unsorted=True,
    )
    left = myMicro.SetFromNodeLabels(
        name = 'Left',
        nodeLabels = tuple(left),
        unsorted=True,
    )
    right = myMicro.SetFromNodeLabels(
        name = 'Right',
        nodeLabels = tuple(right),
        unsorted=True,
    )

    # corners
    topleft = myMicro.SetFromNodeLabels(
        name = 'TopLeft',
        nodeLabels = tuple(topleft),
        unsorted=True,
    )
    topright = myMicro.SetFromNodeLabels(
        name = 'TopRight',
        nodeLabels = tuple(topright),
        unsorted=True,
    )
    botleft = myMicro.SetFromNodeLabels(
        name = 'BottomLeft',
        nodeLabels = tuple(botleft),
        unsorted=True,
    )
    botright = myMicro.SetFromNodeLabels(
        name = 'BottomRight',
        nodeLabels = tuple(botright),
        unsorted=True,
    )

    # Creating the driving node sets

    driving_node_sets = [
            myMicro.SetFromNodeLabels(
            name = 'Driver' + str(n),
            nodeLabels = (drivers[n].label, ),
            unsorted=True,
        ) for n in range(len(drivers))
    ]

    refPoints=(myMicro.referencePoints[2], )
    myMicro.Set(referencePoints=refPoints, name='Driver')

    
    if len(np.unique(hetero_micro)) > 1:
        # Create element set for heterogeneous microstructure
        material_element_sets = [[] for _ in range(len(mech_properties))]
        for element in myMicro.elements:
            centroid = get_centroid(element)
            index = map_centroid_to_index(centroid, micro_shape)
            # append element to cooresponding material element set
            material_element_sets[hetero_micro[index]].append(element)
    
        for n in range(len(material_element_sets)):
            material_element_sets[n] = myMicro.Set(
                name='MatElementSet' + str(n),
                elements=mesh.MeshElementArray(material_element_sets[n]),
            )
        #-------------------------------------------------------
        # Assign Section
    
        for n in range(len(material_element_sets)):
            myMicro.SectionAssignment(
                region = material_element_sets[n],
                sectionName = 'State' + str(n),
                offset = 0.0,
                offsetType=MIDDLE_SURFACE, 
                offsetField='', 
                thicknessAssignment=FROM_SECTION,
            )  
    else: 
        # Create element set for homogeneous microstructure
        single_material_element_set = myMicro.Set(
            name='MatElementSet' + str(int(localstateflag)),
            elements=myMicro.elements,
            )
            
        #-------------------------------------------------------
        # Assign Section
        myMicro.SectionAssignment(
            region = single_material_element_set,
            sectionName = 'State' + str(int(localstateflag)),
            offset = 0.0,
            offsetType=MIDDLE_SURFACE, 
            offsetField='', 
            thicknessAssignment=FROM_SECTION,
        )      

    #-------------------------------------------------------
    import assembly
    # Create a assembly instance.

    myAssembly = myModel.rootAssembly
    # Dependent must be on because we generate the mesh on the part. 
    # Therefore, the Assembly inherits the mesh from the part.
    myInstance = myAssembly.Instance(name='boxAssembly', part=myMicro, dependent=ON)

    #-------------------------------------------------------
    # Define equations for PBC
    # Left-Right
    myModel.Equation(name='rl_1', terms=((1.0, 'boxAssembly.Right', 1), (-1.0, 'boxAssembly.Left', 1), (-1.0, 'boxAssembly.Driver0', 1)))
    myModel.Equation(name='rl_2', terms=((1.0, 'boxAssembly.Right', 2), (-1.0, 'boxAssembly.Left', 2)))

    # Top-Bottom
    myModel.Equation(name='tb_1', terms=((1.0, 'boxAssembly.Top', 1), (-1.0, 'boxAssembly.Bottom', 1), (-1.0, 'boxAssembly.Driver2', 1)))
    myModel.Equation(name='tb_2', terms=((1.0, 'boxAssembly.Top', 2), (-1.0, 'boxAssembly.Bottom', 2), (-1.0, 'boxAssembly.Driver1', 1)))

    # Corner 2 & 1
    myModel.Equation(name='Corner21_1', terms=((-1.0, 'boxAssembly.TopLeft', 1), (1.0, 'boxAssembly.TopRight', 1),
                    (-1.0, 'boxAssembly.Driver0', 1)))
    myModel.Equation(name='Corner21_2', terms=((-1.0, 'boxAssembly.TopLeft', 2), (1.0, 'boxAssembly.TopRight', 2)))

    # Corner 3 & 1
    myModel.Equation(name='Corner31_1', terms=((-1.0, 'boxAssembly.BottomLeft', 1), (1.0, 'boxAssembly.TopRight', 1),
                    (-1.0, 'boxAssembly.Driver0', 1),(-1.0, 'boxAssembly.Driver2', 1)))
    myModel.Equation(name='Corner31_2', terms=((-1.0, 'boxAssembly.BottomLeft', 2), (1.0, 'boxAssembly.TopRight', 2),
                    (-1.0, 'boxAssembly.Driver1', 1)))
                    
    # Corner 4 & 1
    myModel.Equation(name='Corner41_1', terms=((-1.0, 'boxAssembly.BottomRight', 1), (1.0, 'boxAssembly.TopRight', 1),
                    (-1.0, 'boxAssembly.Driver2', 1)))
    myModel.Equation(name='Corner41_2', terms=((-1.0, 'boxAssembly.BottomRight', 2), (1.0, 'boxAssembly.TopRight', 2),
                    (-1.0, 'boxAssembly.Driver1', 1)))
                                
    #-------------------------------------------------------
    # Displacement BC Locking in Corner 1
    myModel.DisplacementBC(
        name='lock_corner1', 
        createStepName='Initial', 
        region=myAssembly.sets['boxAssembly.TopRight'], 
        u1=SET, 
        u2=SET, 
        ur3=SET, 
        amplitude=UNSET, 
        distributionType=UNIFORM, 
        fieldName='', 
        localCsys=None
    )

    #-------------------------------------------------------
    import step
    # Steps are created for loading in X, Y, and XY
    # Boundary conditions for prior steps are deactivated

    for step, boundary_triplet in enumerate(boundary_conditions):
        # run through all of the passed boundary conditions

        myModel.StaticStep(
            name='Step-' + str(step), 
            previous= 'Step-' + str(step-1) if step > 0 else 'Initial',
            timePeriod=1.0, initialInc=0.1
        )
        
        for boundary_index, boundary_value in enumerate(boundary_triplet):
            # run through all of the boundary values and impose them on 
            # the simulation
            #
            # Driver0 controls exx / sxx
            # Driver1 controls eyy / syy
            # Driver2 controls exy / sxy
            #
            # In all cases, control is imposed through the first DOF. 

            # first, we need to deactivate the boundary conditions
            # associated with the previous step. Technically, I think
            # we should be able to just write over them. But, I
            # don't want to risk it. So, instead, I am just going
            # to make new boundary conditions for each additional
            # step
            #
            # BCs will be labeled as XXXX-boundary_index-step. 

            # Abaqus doesn't like you setting boundary values (specifically
            # forces) that are zero. So, thats why we are checking if 
            # things are zero and turning them on and off.

            if step > 0 and \
                np.abs(previous_boundary_triplet[boundary_index]) > eps_bc:
                # then we need to deactivate the previous step's condition
                if plane_stress_flag:
                    # plane stress
                    myModel.loads[
                        'Load-' + str(boundary_index) + '-' + str(step-1)
                    ].deactivate('Step-' + str(step))

                else:
                    # plane strain
                    myModel.boundaryConditions[
                        'BC-' + str(boundary_index) + '-' + str(step-1)
                    ].deactivate('Step-' + str(step))
                    

            if np.abs(boundary_value) > eps_bc:
                if plane_stress_flag:                    
                    # plane stress
                    myModel.ConcentratedForce(
                        name='Load-' + str(boundary_index) + '-' + str(step), 
                        createStepName='Step-' + str(step), 
                        region=myAssembly.sets['boxAssembly.Driver' + str(boundary_index)], 
                        cf1=boundary_value, 
                        distributionType=UNIFORM, 
                        field='', 
                        localCsys=None
                    )
                else:
                    myModel.DisplacementBC(
                        name='BC-' + str(boundary_index) + '-' + str(step), 
                        createStepName='Step-' + str(step), 
                        region=myAssembly.sets['boxAssembly.Driver' + str(boundary_index)], 
                        u1=boundary_value, 
                        u2=0.0, 
                        ur3=0.0, 
                        amplitude=UNSET, 
                        fixed=OFF, 
                        distributionType=UNIFORM, 
                        fieldName='', 
                        localCsys=None
                    )
        
        previous_boundary_triplet = boundary_triplet
                
    # Request specific model outputs
    myModel.fieldOutputRequests['F-Output-1'].setValues(variables=('S', 
        'PE', 'PEEQ', 'PEMAG', 'LE', 'U', 'RF', 'CF', 'CSTRESS', 'CDISP', 'EVOL'))
    
    
    #-------------------------------------------------------
    import job
    # Create an analysis job for the model and submit it.
    description = 'Plane Stress' if plane_stress_flag else 'Plane Strain'
    description = 'Heterogeneous Microstructure -- PBC Homogenization -- ' + description
    
    myJob = mdb.Job(
        name=job_location, 
        model='Box',
        description='Micro PBC Displacement',
        nodalOutputPrecision=SINGLE,
        numCpus = num_cpu_job,
        numDomains = num_cpu_job,
        scratch=scratch_dir,
        resultsFormat=ODB,
        multiprocessingMode=DEFAULT
    )

    # Wait for the job to complete.

    myJob.submit()
    myJob.waitForCompletion()

# Post Processing Scripts
def Plane_Stress_Mechanical_Post(odb_location):    
    '''
    This is the standard post processing method for the
    Heterogeneous 2D Microstructure PBC homogenization call. 
    
    This script assumes that a plane stress simulation was
    performed, that 1 MPa of stress was imposed and that
    three steps were included:

        (1) sxx
        (2) syy
        (3) sxy
        
    This corresponds to the R3 identity being passed as the boundary
    condition in the constructor above. 
    
    :param obd_location: (str) a string containing the location of 
                               the ODB file.
    '''
    #-------------------------------------------------------
    # Post-processing to obtain Ex, Ey, Gxy
    #-------------------------------------------------------
    odb = openOdb(odb_location + '.odb', readOnly=False)
    
    # identifying the node labels associated with the drivers
    lblx = odb.rootAssembly.instances['BOXASSEMBLY'].nodeSets['DRIVER0'].nodes[0].label
    lbly = odb.rootAssembly.instances['BOXASSEMBLY'].nodeSets['DRIVER1'].nodes[0].label
    lblxy = odb.rootAssembly.instances['BOXASSEMBLY'].nodeSets['DRIVER2'].nodes[0].label
    
    # The x-direction tension step
    frametoread=odb.steps['Step-0'].frames[-1]
    
    # extract the average strains (exx, eyy) associated with the 
    # applied tension
    Fx_eps0_x = frametoread.fieldOutputs['U'].values[lblx-1].data[0]
    Fx_eps0_y = frametoread.fieldOutputs['U'].values[lbly-1].data[0]
    
    # 1.0 is used because Fx = 1.0 and length of any side of the domain is 1.0
    # Therefore sigma_xx = Fx / A = 1.0 / 1.0 = 1.0
    E0_x = 1.0/Fx_eps0_x
    v0_xy = -Fx_eps0_y/Fx_eps0_x
    
    print("**************")
    print('<E_x> = ' + str(E0_x))
    print('<v_xy> = ' + str(v0_xy))
    
    # The y-direction tension step
    frametoread=odb.steps['Step-1'].frames[-1]
    
    Fy_eps0_x = frametoread.fieldOutputs['U'].values[lblx-1].data[0]
    Fy_eps0_y = frametoread.fieldOutputs['U'].values[lbly-1].data[0]
    E0_y = 1.0/Fy_eps0_y
    v0_yx = -Fy_eps0_x/Fy_eps0_y
    
    print("**************")
    print('<E_y> = ' + str(E0_y))
    print('<v_yx> = ' + str(v0_yx))
    
    # XY - Shear step
    frametoread=odb.steps['Step-2'].frames[-1]
    Shear_xy_gamma0_xy = frametoread.fieldOutputs['U'].values[lblxy-1].data[0]

    G0_xy = 1.0/Shear_xy_gamma0_xy
    
    print("**************")
    print('<G_xy> = ' + str(G0_xy))
    
    odb.close()
    
    return E0_x, v0_xy, E0_y, v0_yx, G0_xy
    
def PBC_Homogenization_2D_Thermal(
    hetero_micro,
    job_location,
    num_cpu_job,
    boundary_conditions = (
        (1.0, 0.0),
        (0.0, 1.0),
    ),
    thermal_properties = (
        (1,),
        (100,),
    ),    
):
    """
    This method initializes and runs a complete job performing linear
    elastic homogenization using periodic boundary conditions. The 
    conditions are adapted from Li's 2000 paper "General Unit Cells
    for Micromechnical Analyses of Unidirectional Composites". Instead
    of Generalized Plain Strain, we only implement plain strain / stress. 

    The method will run analysis on a passed microstructure and
    save an odb file according to the 'job_location' parameter.

    Args:
        hetero_micro (np.ndarray): A numpy array containing the desired
                        microstructure for analysis. The array should be of 
                        the shape: [SPATIAL_DIMENSIONS x STATES]. States should
                        be encoded in a one-hot fashion (i.e., only one state
                        can exist in each voxel at once). 
        job_location (str): The location where the ODB file should be saved
                                and its name. 
        plane_stress_flag (bool, optional): A flag indicating
                        plane stress or plane strain simulations. Defaults to True.
        boundary_conditions (tuple, optional): An array / tuple containing
                    the desired boundary conditions. Each element in the tuple will be
                    applied as a single step in the simulation.
                    Defaults to ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), ).
        mech_properties (tuple, optional): A tuple of the desired
                    mechanical properties of each constituent. 
                    Assumes that each constituent is a isotropic linear
                    elastic material.
                    Defaults to ( (1.E3, 0.3), (10.E3, 0.3), ).
    """

    
    assert len(hetero_micro.shape) == 3, 'Microstructures are expected to ' + \
        'have 2 spatial dimensions and one state dimension (at the ' + \
            'end). Only eigenmicrostructures are supported.'

    # transform microstructure to numbered encoding -- 
    # i.e., (0, 1, 2, ...) for each state       
    hetero_micro = map_onehot_to_numbered(hetero_micro)
    micro_shape = hetero_micro.shape

    assert micro_shape[0] == micro_shape[1], \
        'This implementation only supports squares.'

    # Defining the element type
    element_types = (DC2D4, DC2D3)
        
    # check if there is only 1 active state and define flag
    if len(np.unique(hetero_micro)) == 1:
        localstateflag = np.unique(hetero_micro)
    
    #-----------------------------------------------------
    # Create a model container. 
    myModel = mdb.Model(name='Box')

    #-----------------------------------------------------
    import part
    # Create a sketch for the base feature.

    mySketch = myModel.ConstrainedSketch(name='Box',
        sheetSize=250.)

    mySketch.rectangle(point1=(0,0), point2=(1,1))

    myMicro = myModel.Part(name='Micro', dimensionality=TWO_D_PLANAR,
        type=DEFORMABLE_BODY)

    # tell it that it will be 2D. 
    myMicro.BaseShell(sketch=mySketch)

    #-----------------------------------------------------
    import material
    # Create a material and sections
    # IF YOU WANT TO IMPLEMENT A DIFFERENT MATERIAL TYPE
    # (e.g., elastic rigid plastic) YOU SHOULD ONLY NEED 
    # TO CHANGE THE FOLLOWING LINES.

    mats = []
    sections = []

    for n, thermal_prop in enumerate(thermal_properties):
        # creating the material
        mats.append(
            myModel.Material(name='Mat' + str(n))
        )
        mats[-1].Conductivity(table=((thermal_prop, ),))

        # creating the section
        sections.append(
            myModel.HomogeneousSolidSection(
                name = 'State' + str(n),
                material='Mat' + str(n),
                thickness=1.0,
            )
        )
     
    #-------------------------------------------------------
    import mesh
    # Mesh the part.
    # e.g., For Plane Strain
    desired_element = mesh.ElemType(elemCode=element_types[0], elemLibrary=STANDARD)
    backup_element = mesh.ElemType(elemCode=element_types[1], elemLibrary=STANDARD)

    faces = myMicro.faces.getSequenceFromMask(mask=('[#1 ]', ),)
    myMicro.setElementType(
        regions=(faces, ), 
        elemTypes=(desired_element, backup_element),
    )

    # Create a mesh with the same number of elements as voxels
    size_value = 1.0 / micro_shape[0]
    myMicro.seedPart(
        size=size_value, 
        deviationFactor=size_value, 
        minSizeFactor=size_value
    )
    myMicro.generateMesh()

    # add DriverNodes
    drivers = []
    drivers = [
        myMicro.Node(coordinates=(0.5, 0.5, 0.0)),
        myMicro.Node(coordinates=(0.5, 0.5, 0.0)),
        myMicro.Node(coordinates=(0.5, 0.5, 0.0)),
    ]

    #--------------------------------------------------------
    # Creating Node and Element Sets

    # Create Node sets for boundary conditions
    all_nodes = myMicro.nodes

    # sides
    top = []
    bottom = []
    left = []
    right = []

    # corners
    topleft = []
    topright = []
    botleft = []
    botright = []

    for node in myMicro.nodes:
        xcoord, ycoord = node.coordinates[:len(micro_shape)]

        if np.abs(xcoord) <= eps:
            # near left edge
            if np.abs(ycoord) <= eps:
                # left bottom corner
                botleft.append(node.label)
            
            elif np.abs(1-ycoord) <= eps:
                # left top corner
                topleft.append(node.label)
            
            else:
                # just on the left edge
                left.append(node.label)

        elif np.abs(1-xcoord) <= eps:
            # near right edge
            if np.abs(ycoord) <= eps:
                # right bottom corner
                botright.append(node.label)
            
            elif np.abs(1-ycoord) <= eps:
                # right top corner
                topright.append(node.label)
            
            else:
                # just on the right edge
                right.append(node.label)
        
        elif np.abs(ycoord) <= eps:
            # bottom -- but no need to check for corners. 
            bottom.append(node.label)
        
        elif np.abs(1-ycoord) <= eps:
            # top -- but no need to check for corners. 
            top.append(node.label)

    # creating the node sets
    # sides
    top = myMicro.SetFromNodeLabels(
        name = 'Top',
        nodeLabels = tuple(top),
        unsorted=True,
    )
    bottom = myMicro.SetFromNodeLabels(
        name = 'Bottom',
        nodeLabels = tuple(bottom),
        unsorted=True,
    )
    left = myMicro.SetFromNodeLabels(
        name = 'Left',
        nodeLabels = tuple(left),
        unsorted=True,
    )
    right = myMicro.SetFromNodeLabels(
        name = 'Right',
        nodeLabels = tuple(right),
        unsorted=True,
    )

    # corners
    topleft = myMicro.SetFromNodeLabels(
        name = 'TopLeft',
        nodeLabels = tuple(topleft),
        unsorted=True,
    )
    topright = myMicro.SetFromNodeLabels(
        name = 'TopRight',
        nodeLabels = tuple(topright),
        unsorted=True,
    )
    botleft = myMicro.SetFromNodeLabels(
        name = 'BottomLeft',
        nodeLabels = tuple(botleft),
        unsorted=True,
    )
    botright = myMicro.SetFromNodeLabels(
        name = 'BottomRight',
        nodeLabels = tuple(botright),
        unsorted=True,
    )

    # Creating the driving node sets

    driving_node_sets = [
            myMicro.SetFromNodeLabels(
            name = 'Driver' + str(n),
            nodeLabels = (drivers[n].label, ),
            unsorted=True,
        ) for n in range(len(drivers))
    ]
    
    if len(np.unique(hetero_micro)) > 1:
        # Create element set for heterogeneous microstructure
        material_element_sets = [[] for _ in range(len(thermal_properties))]
        for element in myMicro.elements:
            centroid = get_centroid(element)
            index = map_centroid_to_index(centroid, micro_shape)
            # append element to cooresponding material element set
            material_element_sets[hetero_micro[index]].append(element)
    
        for n in range(len(material_element_sets)):
            material_element_sets[n] = myMicro.Set(
                name='MatElementSet' + str(n),
                elements=mesh.MeshElementArray(material_element_sets[n]),
            )
        #-------------------------------------------------------
        # Assign Section
    
        for n in range(len(material_element_sets)):
            myMicro.SectionAssignment(
                region = material_element_sets[n],
                sectionName = 'State' + str(n),
                offset = 0.0,
                offsetType=MIDDLE_SURFACE, 
                offsetField='', 
                thicknessAssignment=FROM_SECTION,
            )  
    else: 
        # Create element set for homogeneous microstructure
        single_material_element_set = myMicro.Set(
            name='MatElementSet' + str(int(localstateflag)),
            elements=myMicro.elements,
            )
            
        #-------------------------------------------------------
        # Assign Section
        myMicro.SectionAssignment(
            region = single_material_element_set,
            sectionName = 'State' + str(int(localstateflag)),
            offset = 0.0,
            offsetType=MIDDLE_SURFACE, 
            offsetField='', 
            thicknessAssignment=FROM_SECTION,
        )      

    #-------------------------------------------------------
    import assembly
    # Create a assembly instance.

    myAssembly = myModel.rootAssembly
    # Dependent must be on because we generate the mesh on the part. 
    # Therefore, the Assembly inherits the mesh from the part.
    myInstance = myAssembly.Instance(name='boxAssembly', part=myMicro, dependent=ON)

    #-------------------------------------------------------
    # Define equations for PBC
    # Left-Right
    myModel.Equation(name='rl_1', terms=((1.0, 'boxAssembly.Right', 11), (-1.0, 'boxAssembly.Left', 11), (-1.0, 'boxAssembly.Driver0', 11)))

    # Top-Bottom
    myModel.Equation(name='tb_1', terms=((1.0, 'boxAssembly.Top', 11), (-1.0, 'boxAssembly.Bottom', 11), (-1.0, 'boxAssembly.Driver1', 11)))

    # Corner 2 & 1
    myModel.Equation(name='Corner21_1', terms=((-1.0, 'boxAssembly.TopLeft', 11), (1.0, 'boxAssembly.TopRight', 11),
                    (-1.0, 'boxAssembly.Driver0', 11)))

    # Corner 3 & 1
    myModel.Equation(name='Corner31_1', terms=((-1.0, 'boxAssembly.BottomLeft', 11), (1.0, 'boxAssembly.TopRight', 11),
                    (-1.0, 'boxAssembly.Driver0', 11),(-1.0, 'boxAssembly.Driver1', 11)))
    
    # Corner 4 & 1
    myModel.Equation(name='Corner41_1', terms=((-1.0, 'boxAssembly.BottomRight', 11), (1.0, 'boxAssembly.TopRight', 11),
                    (-1.0, 'boxAssembly.Driver1', 11)))
                    
    #-------------------------------------------------------
    import step
    # Steps are created for loading in X, Y
    # Boundary conditions for prior steps are deactivated

    for step, boundary_double in enumerate(boundary_conditions):
        # run through all of the passed boundary conditions     

        myModel.HeatTransferStep(
            name='Step-' + str(step), 
            previous= 'Step-' + str(step-1) if step > 0 else 'Initial',
            response=STEADY_STATE, amplitude=RAMP
        )
        
        for boundary_index, boundary_value in enumerate(boundary_double):
            # run through all of the boundary values and impose them on 
            # the simulation
            #
            # Driver0 controls qx      
            # Driver1 controls qy
            #
            # In all cases, control is imposed through the 11 DOF. 

            # first, we need to deactivate the boundary conditions
            # associated with the previous step. Technically, I think
            # we should be able to just write over them. But, I
            # don't want to risk it. So, instead, I am just going
            # to make new boundary conditions for each additional
            # step
            #
            # BCs will be labeled as XXXX-boundary_index-step. 

            # Abaqus doesn't like you setting boundary values (specifically
            # forces) that are zero. So, thats why we are checking if 
            # things are zero and turning them on and off.

            if step > 0 and \
                np.abs(previous_boundary_double[boundary_index]) > eps_bc:
                # then we need to deactivate the previous step's condition
                myModel.loads[
                    'Load-' + str(boundary_index) + '-' + str(step-1)
                ].deactivate('Step-' + str(step))

            if np.abs(boundary_value) > eps_bc:                 
                myModel.ConcentratedHeatFlux(
                    name='Load-' + str(boundary_index) + '-' + str(step), 
                    createStepName='Step-' + str(step), 
                    region=myAssembly.sets['boxAssembly.Driver' + str(boundary_index)], 
                    magnitude=1.0,
                )

        previous_boundary_double = boundary_double
             
    #-------------------------------------------------------
    import job
    # Create an analysis job for the model and submit it.
    description = 'Heterogeneous Microstructure -- PBC Homogenization -- '
    
    myJob = mdb.Job(
        name=job_location, 
        model='Box',
        description='Micro PBC Displacement',
        nodalOutputPrecision=SINGLE,
        numCpus = num_cpu_job,
        numDomains = num_cpu_job,
        scratch='',
        resultsFormat=ODB,
        multiprocessingMode=DEFAULT
    )

    # Wait for the job to complete.

    myJob.submit()
    myJob.waitForCompletion()

# Post Processing Scripts
def Thermal_Post(odb_location):    
    odb = openOdb(odb_location + '.odb', readOnly=False)
    
    # identifying the node labels associated with the drivers
    lblx = odb.rootAssembly.instances['BOXASSEMBLY'].nodeSets['DRIVER0'].nodes[0].label
    lbly = odb.rootAssembly.instances['BOXASSEMBLY'].nodeSets['DRIVER1'].nodes[0].label
    
    # The x-direction heat flux step
    frametoread=odb.steps['Step-0'].frames[-1]
    
    # extract the average temperature gradient associated with the 
    # applied heat flux
    Fx_eps0_x = frametoread.fieldOutputs['NT11'].values[lblx-1].data
    
    # 1.0 is used because qx = 1.0 and length of any side of the domain is 1.0
    # Therefore grad T_xx = qx / L = 1.0 / 1.0 = 1.0
    k0_x = 1.0/Fx_eps0_x
    
    print("**************")
    print('<k_x> = ' + str(k0_x))
    
    # The y-direction tension step
    frametoread=odb.steps['Step-1'].frames[-1]
    
    Fy_eps0_y = frametoread.fieldOutputs['NT11'].values[lbly-1].data
    k0_y = 1.0/Fy_eps0_y
    
    print("**************")
    print('<k_y> = ' + str(k0_y))    
    
    odb.close()
    
    return k0_x, k0_y

