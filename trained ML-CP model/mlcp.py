#!/home/jvanbuskirk/Python-3.8.5/bin/python3.8

def print_cif(cif, sf=1.0, temp=None):
    import time
    t1 = time.time()
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn
    #from pyodide.ffi import to_js
    import pickle
    import math
    from scipy.spatial.distance import cdist
    from itertools import combinations
    import numpy as np
    from pymatgen.core.structure import Structure, Lattice, IStructure
    from pymatgen.io.xyz import XYZ
    from pymatgen.analysis.local_env import VoronoiNN
    from pymatgen.core import periodic_table as PT

    radius = 5.0

    def modifile(filename, action, lines=None):
        # Reads, writes, or appends a file
        if action == "r":
            with open(filename, "r") as file:
                lines = file.readlines()
            return lines
        else:
            if lines is not None:
                with open(filename, action) as file:
                    file.writelines(lines)
                return 1
            else:
                return 0

    #modifile("cif.cif", "w", cif)
    df2 = modifile("./element_data.txt", "r")
    for i in range(len(df2)):
        df2[i] = [float(j) for j in df2[i].split("\t")]

    structure = Structure.from_file(cif)
    structure.scale_lattice(structure.volume*(float(sf)**3))
    for i in range(len(structure)):
        c = structure[i].frac_coords
        for j in range(3):
            if c[j] > -1e-5 and c[j] < -1e-25:
                structure[i].frac_coords[j] = 0
            if c[j] == 1:
                structure[i].frac_coords[j] = 0
            if c[j] < -1e-5 and c[j] > -1e-1:
                structure[i].frac_coords[j] = c[j]+1
            if c[j] < 1.1 and c[j] > 1:
                structure[i].frac_coords[j] = c[j]-1

    cell = [list(i) for i in structure.lattice.matrix]
    summary = str(structure).split("\n")[:4]

    if temp != None:
        #modifile("cif.xyz", "w", temp)
        xyz = XYZ.from_file(temp).as_dataframe()
        species = xyz['atom'].tolist()
        x = xyz['x'].tolist()
        y = xyz['y'].tolist()
        z = xyz['z'].tolist()
        xyzs = [[x[i], y[i], z[i]] for i in range(len(x))]
        xyzs_template = [[species[i], x[i], y[i], z[i]] for i in range(len(x))]
        substructure = Structure(structure.lattice, species, xyzs, coords_are_cartesian=True)

    t2 = time.time()

    hkls = []
    for h in range(-1,2):
        for k in range(-1,2):
            for l in range(-1,2):
                hkls.append([h,k,l])
    coord_shifts = []
    cell_array = np.array(cell)
    for k in range(len(hkls)):
        coord_shifts.append(np.dot(hkls[k], cell_array))
    coord_shifts = np.array(coord_shifts)

    sup = structure.copy()
    sup.make_supercell(2)
    nn = structure.get_all_neighbors(radius)
    # get voronoi data for all relevant atoms
    if temp != None:
        allvnn = []
        getvoro = [i for i in xyzs]
        for i in range(len(substructure)):
            for j in range(len(structure)):
                test_struct = np.transpose(np.repeat(structure[j].coords, 27).reshape(3, 27))
                test_sub = np.transpose(np.repeat(substructure[i].coords, 27).reshape(3, 27))
                sub = test_struct - test_sub + coord_shifts
                w = np.where(np.bincount(np.where(np.logical_and(sub < 0.001, sub > -0.001))[0])==3)[0]
                if len(w) == 1:
                    for k in range(len(nn[j])):
                        getvoro.append(nn[j][k].coords)
                    break
        getvoro = np.unique(np.array(getvoro).round(decimals=6), axis=0)
        getvoro_frac = np.dot(getvoro, np.linalg.inv(np.array(cell)))
        getvoro_where = np.where(np.logical_and(getvoro_frac>-0.001, getvoro_frac<1), 1, 0)
        getvoro_in_uc = np.where(np.sum(getvoro_where, axis=1)==3)[0]
        getvoro = np.unique(getvoro[getvoro_in_uc], axis=0)
        for i in range(len(structure)):
            for j in range(len(getvoro)):
                if np.all(np.isclose(getvoro[j], structure[i].coords)):
                    allvnn.append([value for key, value in VoronoiNN().get_voronoi_polyhedra(structure,i).items()])
                    break
            else:
                allvnn.append([])
    else:
        allvnn = VoronoiNN().get_all_nn_info(structure)

    nn_options = []
    neighbor_weights = []
    neighbors = []
    neighbors2 = []
    cns = []
    stoich = []
    elems = []
    data = []
    used_for = []
    vectors = []
    geo = []
    geo2 = []
    struct_coords = []
    template = []
    template2 = []
    template3 = []
    vols = []
    s_tot = []
    en = []
    rad = []
    en2 = []
    rad2 = []
    v = 0

    for i in range(len(allvnn)):
        if structure[i].specie.Z in stoich:
            s_tot[stoich.index(structure[i].specie.Z)] += 1
            en.append(PT.Element(elems[-1]).X**3)
            rad.append(PT.Element(elems[-1]).atomic_radius_calculated**3)
        else:
            stoich.append(structure[i].specie.Z)
            s_tot.append(1)
            elems.append(structure[i].specie.symbol)
            en.append(PT.Element(elems[-1]).X**3)
            rad.append(PT.Element(elems[-1]).atomic_radius_calculated**3)
    ensum = sum(en)/len(en)
    radsum = sum(rad)/len(rad)
    for i in range(len(en)):
        en2.append(en[i]/ensum)
        rad2.append(rad[i]/radsum)

    for i in range(len(allvnn)):
        neighbors.append([])
        neighbors2.append([])
        vols.append([])
        cns.append([0,0])
        for j in range(len(allvnn[i])):
            neighbors2[i].append(allvnn[i][j]['site'].to_unit_cell())
            neighbors[i].append(allvnn[i][j]['site'])
            if temp != None:
                vols[i].append(round(allvnn[i][j]['volume'], 6))
            else:
                vols[i].append(round(allvnn[i][j]['poly_info']['volume'], 6))
            if allvnn[i][j]['site'].specie.Z == structure[i].specie.Z:
                cns[i][0] += 1
            else:
                cns[i][1] += 1

    t3 = time.time()

    sn = sup.get_all_neighbors(radius)

    dx = 2 # search x angstroms beyond the unit cell.
    def within_range(d, dx, a, b, c):
        if -dx <= d[0] and d[0] <= a+dx:
            if -dx <= d[1] and d[1] <= b+dx:
                if -dx <= d[2] and d[2] <= c+dx:
                    return(True)
                else:
                    return(False)
            else:
                return(False)
        else:
            return(False)

    for i in range(len(sup)):
        a = structure.lattice.a
        b = structure.lattice.b
        c = structure.lattice.c
        d = sup[i].coords
        for j in range(len(sn[i])):
            d = sn[i][j].coords
            if within_range(d, dx, a, b, c):
                template.append([sn[i][j].specie.symbol, d[0], d[1], d[2]])

    [template2.append(x) for x in template if x not in template2]
    template3 = template2.copy()

    for i in range(len(structure)):
        s = structure[i]
        if s.specie.Z in stoich:
            s_tot[stoich.index(s.specie.Z)] += 1
        else:
            stoich.append(s.specie.Z)
            s_tot.append(1)
            elems.append(s.specie.symbol)
        geo.append([s.specie.symbol, round(s.coords[0],4), round(s.coords[1],4), round(s.coords[2],4)])
        geo2.append([round(s.frac_coords[0],4), round(s.frac_coords[1],4), round(s.frac_coords[2],4)])
        struct_coords.append(s.coords)
    struct_coords = np.array(struct_coords)

    cell_voxel = np.array([np.array(i, dtype=np.float16) for i in cell])
    positions = []
    x = []
    y = []
    z = []

    for i in range(len(geo2)):
        positions.append(np.dot(geo2[i], cell_voxel))

    if temp != None:
        coords = np.array([i.frac_coords for i in substructure])
        mincv = np.min(coords, axis=0)
        maxcv = np.max(coords, axis=0)
        for i in range(3):
            if mincv[i] < 0:
                mincv[i] = 0
            if maxcv[i] > 1:
                maxcv[i] = 1
        coords_first_uc = []
        for i in range(len(xyzs)):
            c = coords[i]
            for j in range(3):
                if c[j] < 0:
                    c[j] = c[j] + 1
                if c[j] >= 1:
                    c[j] = c[j] - 1
                if c[j] < 1e-5:
                    c[j] = 0
            coords_first_uc.append([round(c[0],4), round(c[1],4), round(c[2],4)])
        ucfuc = []
        [ucfuc.append(p) for p in coords_first_uc if p not in ucfuc]
        ucfuc = np.array(ucfuc)
        xyzs_first_uc = np.dot(ucfuc, cell_voxel)
        xyzs_first_uc = np.array([[round(i[0],4), round(i[1],4), round(i[2],4)] for i in xyzs_first_uc])
        for i in range(len(xyzs_first_uc)):
            x.append(xyzs_first_uc[i][0])
            y.append(xyzs_first_uc[i][1])
            z.append(xyzs_first_uc[i][2])
        vca_small = []
        vca_weights_small = []
        cubic_grid_small = []
        filter_indicies = []
        geo_list = [list(l) for l in geo2]
        ucfuc_list = [list(l) for l in ucfuc]
        for i in range(len(ucfuc_list)):
            if ucfuc_list[i] in geo_list:
                filter_indicies.append(geo_list.index(ucfuc_list[i]))
    else:
        for i in range(len(positions)):
            x.append(positions[i][0])
            y.append(positions[i][1])
            z.append(positions[i][2])
        filter_indicies = [i for i in range(len(structure))]

    unique = []
    el1_data = df2[stoich[0]]
    el2_data = df2[stoich[1]]
    e_data = []
    for l in range(14):
        e_data.append((el1_data[l]*s_tot[0] + el2_data[l]*s_tot[1])/(s_tot[0]+s_tot[1]))
        e_data.append(np.sqrt((s_tot[0]*(el1_data[l]-e_data[-1])**2 + (s_tot[1]*(el2_data[l]-e_data[-1])**2))/(s_tot[0]+s_tot[1])))
    for i in range(len(structure)):
        if i in filter_indicies:
            for j in range(len(neighbors[i])):
                ct1_data = df2[structure[i].specie.Z]
                ct2_data = df2[neighbors[i][j].specie.Z]
                length = round(np.linalg.norm(structure[i].coords - neighbors[i][j].coords), 4)
                test = np.transpose(np.repeat(neighbors2[i][j].coords, len(structure)).reshape(3, len(structure)))
                sub = test - struct_coords
                k = np.where(np.bincount(np.where(np.isclose(sub,np.zeros(sub.shape),rtol=0.001,atol=0.000001))[0])==3)[0][0]
                cns_neighbor = cns[k]
                sc = structure[i].frac_coords.tolist()
                nc = neighbors[i][j].frac_coords.tolist()
                if [i, k, sc, nc] in unique or [k, i, nc, sc] in unique:
                    continue
                else:
                    vven = round(vols[i][j]*(en2[i]+en2[k]), 4)
                    neighbor_weights.append(vven*0.1*(1/length))
                    is_same = 0 if (neighbors2[i][j].coords == neighbors[i][j].coords).all() else 1
                    sum_met_rad = (ct1_data[2] + ct2_data[2])/100
                    sum_met_vol = round((ct1_data[2]/100)**3 + (ct2_data[2]/100)**3, 4)
                    length_cubed = round(length**3, 4)
                    en_range = round(abs(ct1_data[6] - ct2_data[6]), 4)
                    diff_len = round(sum_met_rad - length, 4)
                    diff_vorovol = round(vven - length_cubed, 4)
                    diff_metvol = round(sum_met_vol - length_cubed, 4)
                    extras = [length, vven, sum_met_rad, sum_met_vol, length_cubed, en_range, diff_len, diff_vorovol, diff_metvol]
                    c_data = []
                    cns_data = []
                    for l in range(14):
                        c_data.append((ct1_data[l]+ct2_data[l])/2)
                        c_data.append(abs(ct1_data[l]-ct2_data[l])/2)
                    for l in range(2):
                        cns_data.append((cns[i][l]+cns_neighbor[l])/2)
                        cns_data.append(abs(cns[i][l]+cns_neighbor[l])/2)
                    data.append(extras + e_data + c_data + cns_data)
                    vectors.append(neighbors[i][j].coords - structure[i].coords)
                    if is_same == 0:
                        used_for.append([i,k])
                    else:
                        used_for.append([i,len(structure)])
                    unique.append([i, k, structure[i].frac_coords.tolist(), neighbors[i][j].frac_coords.tolist()])

    npdata = np.array([np.array(line) for line in data])
    npdata3 = [list(a) for a in npdata]
    assignments = []
    unique_npdata = []
    for i in range(len(npdata3)):
        if npdata3[i] not in unique_npdata:
            if npdata3[i][:28]+npdata3[i][42:56]+npdata3[i][28:42]+npdata3[i][56:] not in unique_npdata:
                unique_npdata.append(npdata3[i])
                assignments.append(len(unique_npdata)-1)
            else:
                assignments.append(unique_npdata.index(npdata3[i][:28]+npdata3[i][42:56]+npdata3[i][28:42]+npdata3[i][56:]))
        else:
            assignments.append(unique_npdata.index(npdata3[i]))
    unique_npdata = [np.array(a) for a in unique_npdata]

    f = open('./rfc.pickle', 'rb')
    rfc = pickle.load(f)
    f.close()

    pred_rfc = rfc.predict(unique_npdata)
    predictions_rfc = np.reshape(pred_rfc, (pred_rfc.shape[0],1))
    npdata2 = np.hstack((unique_npdata, predictions_rfc))

    rfc = 0

    f = open('./rfr.pickle', 'rb')
    rfr = pickle.load(f)
    f.close()

    unique_predictions_rfr = rfr.predict(npdata2)

    rfr = 0

    predictions_rfr = []
    for i in range(len(assignments)):
        predictions_rfr.append(unique_predictions_rfr[assignments[i]])
    predictions_rfr = np.array(predictions_rfr)
    geo2 = np.array([np.array(i) for i in geo2])
    neighborr = unique.copy()

    t4 = time.time()

    for j in range(len(neighborr)):
        for i in range(3):
            if neighborr[j][2][i] >= 0 and neighborr[j][2][i] < 1:
                neighborr[j][2][i] = 0
            elif neighborr[j][2][i] >= -1 and neighborr[j][2][i] < 0:
                neighborr[j][2][i] = -1
            else: # neighbor should be greater than one
                neighborr[j][2][i] = 1
            if neighborr[j][3][i] >= 0 and neighborr[j][3][i] < 1:
                neighborr[j][3][i] = 0
            elif neighborr[j][3][i] >= -1 and neighborr[j][3][i] < 0:
                neighborr[j][3][i] = -1
            else: # neighbor should be greater than one
                neighborr[j][3][i] = 1

    contact_verts = []
    contact_atoms = []

    buffer = 7
    voxels_per_angstrom = 5

    def make_cubic_grid(voxels_per_angstrom, mins=[0,0,0], maxs=[1,1,1], buff=0):
        # determine the cell vector lengths
        lena = np.sqrt(cell_voxel[0][0]**2 + cell_voxel[0][1]**2 + cell_voxel[0][2]**2)
        lenb = np.sqrt(cell_voxel[1][0]**2 + cell_voxel[1][1]**2 + cell_voxel[1][2]**2)
        lenc = np.sqrt(cell_voxel[2][0]**2 + cell_voxel[2][1]**2 + cell_voxel[2][2]**2)
        # calculate the length needed for increments, which is (max-min) + 2 * buffer, or len, whichever is less.
        lengths = [lena, lenb, lenc]
        start = [0,0,0]
        stop = [1,1,1]
        for i in range(3):
            # along a, b, and c, if the substructure is entirely contained within the unit cell, we don't need the whole range.
            # note: if the substructure lies along the unit cell edge on one side, but not the other, we can't do this.
            if mins[i]-(buff/lengths[i]) > 0:
                if maxs[i]+(buff/lengths[i]) < 1:
                    start[i] = mins[i]-(buff/lengths[i])
                    stop[i] = maxs[i]+(buff/lengths[i])
            if lengths[i] > ((maxs[i]-mins[i]) * lengths[i] + buff * 2):
                lengths[i] = ((maxs[i]-mins[i]) * lengths[i] + buff * 2)
        # calculate the number of increments along each vector
        na = int(math.ceil(lengths[0] * voxels_per_angstrom)) + 1
        nb = int(math.ceil(lengths[1] * voxels_per_angstrom)) + 1
        nc = int(math.ceil(lengths[2] * voxels_per_angstrom)) + 1
        na_max = int(math.ceil(lena * voxels_per_angstrom))
        nb_max = int(math.ceil(lenb * voxels_per_angstrom))
        nc_max = int(math.ceil(lenc * voxels_per_angstrom))
        # create the fractional coordinates along each vector
        a_space = np.linspace(start[0], stop[0], na, dtype="float16")
        b_space = np.linspace(start[1], stop[1], nb, dtype="float16")
        c_space = np.linspace(start[2], stop[2], nc, dtype="float16")
        # center the coordinates (ex.: [0.0, 0.1, ... 0.8, 0.9] ==> [0.05, 0.015, ... 0.85, 0.95]
        a_space = (a_space[:-1] + a_space[1:]) / 2  # gets centers of the voxels
        b_space = (b_space[:-1] + b_space[1:]) / 2  # gets centers of the voxels
        c_space = (c_space[:-1] + c_space[1:]) / 2  # gets centers of the voxels
        # combine a, b, and c coordinates
        grid_positions = np.array([np.array([x, y, z], dtype="float16") for x in a_space for y in b_space for z in c_space])
        return(np.dot(grid_positions, cell_voxel), [na_max, nb_max, nc_max])

    if temp != None:
        cubic_grid,voxel_max = make_cubic_grid(voxels_per_angstrom, mincv, maxcv, buffer)
    else:
        cubic_grid,voxel_max = make_cubic_grid(voxels_per_angstrom)
    vmt = voxel_max[0]*voxel_max[1]*voxel_max[2]
    #vca = [] # added as [index of p1, index of p2, [h,k,l] of p1, [h,k,l] of p2]

    offset = []
    p_test_hkl = []
    p_test_xyz = []
    p_test_hkl2 = []
    p_test_xyz2 = []
    p_test_hkl3 = []
    p_test_xyz3 = []
    for i in range(len(positions)):
        for h in range(-2,3):
            for k in range(-2,3):
                for l in range(-2,3):
                    p_test = []
                    for j in range(3):
                        p_test.append(positions[i][j] + cell_voxel[0][j]*h + cell_voxel[1][j]*k + cell_voxel[2][j]*l)
                    p_test_hkl2.append([i,h,k,l])
                    p_test_xyz2.append(p_test)

    t5 = time.time()

    minmax = [[min(x)-buffer, max(x)+buffer],[min(y)-buffer, max(y)+buffer],[min(z)-buffer, max(z)+buffer]]

    for i in range(len(p_test_hkl2)):
        pp = p_test_xyz2[i]
        if pp[0] > minmax[0][0] and pp[0] < minmax[0][1]:
            if pp[1] > minmax[1][0] and pp[1] < minmax[1][1]:
                if pp[2] > minmax[2][0] and pp[2] < minmax[2][1]:
                    p_test_hkl3.append(p_test_hkl2[i])
                    p_test_xyz3.append(np.array(p_test_xyz2[i], dtype=np.float32))

    if temp != None:
        # remove atoms from superstructure which are more than <buffer> angstroms away from any one atom in the substructure
        distances = cdist(p_test_xyz3, xyzs_first_uc)
        mins = np.min(distances, axis=1)
        maybe = np.where(mins<buffer,1,0)
        for i in range(len(p_test_hkl3)):
            if maybe[i] == 1:
                p_test_hkl.append(p_test_hkl3[i])
                p_test_xyz.append(p_test_xyz3[i])
    else:
        p_test_xyz = p_test_xyz3
        p_test_hkl = p_test_hkl3

    p_test_xyz = np.array(p_test_xyz, dtype="float32")

    if temp != None:
        cell_voxel = np.array(cell_voxel, dtype="float32")
        newmin = np.dot(np.min(p_test_xyz, axis=0), np.linalg.inv(cell_voxel))
        newmax = np.dot(np.max(p_test_xyz, axis=0), np.linalg.inv(cell_voxel))
        for i in range(3):
            if newmin[i] < 0:
                newmin[i] = 0
            if newmax[i] > 1:
                newmax[i] = 1
        newmin = np.dot(newmin, cell_voxel)
        newmax = np.dot(newmax, cell_voxel)
        august = np.where(np.logical_and(cubic_grid > newmin, cubic_grid < newmax))[0]
        sept = np.bincount(august)
        octo = np.where(sept==3)[0]
        new_grid = cubic_grid[octo]
        cubic_grid = new_grid

    q1 = int(len(cubic_grid)/4)
    q2 = int(len(cubic_grid)/2)
    q3 = int(3*len(cubic_grid)/4)

    summ1 = cdist(cubic_grid[:q1],p_test_xyz)
    summ2 = cdist(cubic_grid[q1:q2],p_test_xyz)
    summ3 = cdist(cubic_grid[q2:q3],p_test_xyz)
    summ4 = cdist(cubic_grid[q3:],p_test_xyz)
    summ5 = np.concatenate((summ1,summ2), axis=0, dtype="float16")
    summ1, summ2 = [], []
    summ6 = np.concatenate((summ3,summ4), axis=0, dtype="float16")
    summ3, summ4 = [], []
    summ = np.concatenate((summ5,summ6),axis=0)

    fs = np.min(summ, axis=1)
    ss = np.partition(summ, 1)[:,1] + 1/voxels_per_angstrom

    vca = []
    vca_weights = []
    cubic_grid_big = []

    apv = 0.5/voxels_per_angstrom
    summlen = len(summ[0])
    for i in range(len(summ)):
        bi = np.where(summ[i] <= ss[i])[0]
        b = summ[i][bi]
        c = [a+b for a,b in combinations(b,2)]
        ci = [[a,b] for a,b in combinations(bi,2)]
        d = min(c) + apv
        e = [ci[j] for j in range(len(c)) if c[j] <= d]
        for j in range(len(e)):
            p1 = p_test_hkl[e[j][0]]
            p2 = p_test_hkl[e[j][1]]
            vca.append([p1[0]] + [p2[0]] + p1[1:] + p2[1:])
            cubic_grid_big.append(cubic_grid[i])
            vca_weights.append(1/len(e))

    summ = []

    t6 = time.time()

    unique_contacts = []
    unique_contacts_array = []
    voxel_groups = []
    uca, reconstruct = np.unique(np.array(vca), axis=0, return_inverse=True)

    for i in range(len(uca)):
        voxel_groups.append(np.where(reconstruct == i)[0])
        unique_contacts.append([uca[i][0], uca[i][1], list(uca[i][2:5]), list(uca[i][5:])])
        unique_contacts_array.append(np.concatenate((uca[i][0], uca[i][1], uca[i][2:5], uca[i][5:]), axis=None))

    unique_contacts2 = unique_contacts.copy()
    voxel_groups2 = voxel_groups.copy()

    hklsnp = np.array(hkls)

    uc2 = np.array([unique_contacts[i][2] for i in range(len(unique_contacts))])
    uc3 = np.array([unique_contacts[i][3] for i in range(len(unique_contacts))])

    uc_options = np.array([np.concatenate((unique_contacts[i][:2], uc2[i]+j, uc3[i]+j), axis=None)for i in range(len(unique_contacts)) for j in hklsnp])

    paired_groups = []
    paired_groups_hkl = []

    for i in range(len(unique_contacts)):
        test = np.transpose(np.repeat(unique_contacts_array[i], 27*len(unique_contacts)).reshape(8, 27*len(unique_contacts)))
        sub = test - uc_options
        w = np.where(np.bincount(np.where(sub==0)[0])==8)[0]
        paired_groups.append(w//27)
        paired_groups_hkl.append(hklsnp[w%27])

    for i in range(len(paired_groups)):
        new = []
        newhkl = []
        for j in range(len(paired_groups[i])):
            if paired_groups[i][j] not in new:
                new.append(paired_groups[i][j])
                newhkl.append(paired_groups_hkl[i][j])
        paired_groups[i] = new
        paired_groups_hkl[i] = newhkl

    unique_pg = []
    unique_pg_hkl = []
    for i in range(len(paired_groups)):
        pg = sorted(paired_groups[i])
        if pg not in unique_pg:
            unique_pg.append(paired_groups[i])
            unique_pg_hkl.append(paired_groups_hkl[i])

    unique_contacts3 = []
    voxel_groups3 = []

    for i in range(len(unique_pg)):
        upgi = unique_pg[i][0]
        voxel_groups3.append(voxel_groups[upgi])
        unique_contacts3.append(unique_contacts[upgi])
        if len(unique_pg[i]) > 1:
            for j in range(1,len(unique_pg[i])):
                upgij = unique_pg[i][j]
                hkl = unique_pg_hkl[i][j]
                vg = voxel_groups[upgij]
                grid_shift = np.dot(hkl, cell)
                points = []
                weights = []
                l = len(cubic_grid_big)
                count = 0
                for k in range(len(vg)):
                    points.append(cubic_grid_big[vg[k]])
                    weights.append(vca_weights[vg[k]])
                    voxel_groups3[-1] = np.append(voxel_groups3[-1], l+count)
                    count += 1
                cubic_grid_big = np.append(cubic_grid_big, points + grid_shift, axis=0)
                vca_weights = vca_weights + weights

    uc = np.array([[0,0,0,0,0,0,0,0]])
    for i in range(len(unique_contacts3)):
        u = unique_contacts3[i]
        uc = np.vstack([uc, np.concatenate((np.array(u[0:2]), np.array(u[2]), np.array(u[3])))])
    uc = uc[1:]

    voxel_groups4 = voxel_groups3.copy()
    unique_contacts4 = unique_contacts3.copy()

    for i in range(len(neighborr)):
        if neighborr[i] in unique_contacts3:
            pass
        elif [neighborr[i][1], neighborr[i][0], neighborr[i][3], neighborr[i][2]] in unique_contacts3:
            pass
        else: # quite often, a symmetrically equivalent bond has been found. For instance, missing the [0,4,[0,0,0],[0,0,-1]], but we have the [0,4,[0,0,1],[0,0,0]]
            x = neighborr[i]
            y = np.array(neighborr[i][2])
            z = np.array(neighborr[i][3])
            for hkl in hkls:
                s1 = np.concatenate((np.array([x[0], x[1]]), y+hkl, z+hkl))
                w = np.bincount(np.where(uc==s1)[0])
                try:
                    a = max(w)
                    if max(w) == 8:
                        m = np.argmax(w)
                        vg = voxel_groups3[m]
                        unique_contacts4.append(x)
                        voxel_groups4.append([])
                        grid_shift = -np.dot(hkl, cell_voxel)
                        points = []
                        weights = []
                        l = len(cubic_grid_big)
                        count = 0
                        for j in range(len(vg)):
                            points.append(cubic_grid_big[vg[j]])
                            weights.append(vca_weights[vg[j]])
                            voxel_groups4[-1].append(l+count)
                            count += 1
                        cubic_grid_big = np.append(cubic_grid_big, points + grid_shift, axis=0)
                        vca_weights = vca_weights + weights
                        break
                except:
                    pass
                s2 = np.concatenate((np.array([x[1], x[0]]), z+hkl, y+hkl))
                w = np.bincount(np.where(uc==s2)[0])
                try:
                    if max(w) == 8:
                        m = np.argmax(w)
                        vg = voxel_groups3[m]
                        unique_contacts4.append(x)
                        voxel_groups4.append([])
                        grid_shift = -np.dot(hkl, cell_voxel)
                        points = []
                        weights = []
                        l = len(cubic_grid_big)
                        count = 0
                        for j in range(len(vg)):
                            points.append(cubic_grid_big[vg[j]])
                            weights.append(vca_weights[vg[j]])
                            voxel_groups4[-1].append(l+count)
                            count += 1
                        cubic_grid_big = np.append(cubic_grid_big, points + grid_shift, axis=0)
                        vca_weights = vca_weights + weights
                        break
                except:
                    pass

    voxel_groups5 = []
    unique_contacts5 = []
    for i in range(len(unique_contacts4)):
        if unique_contacts4[i] in neighborr:
            unique_contacts5.append(unique_contacts4[i])
            voxel_groups5.append(voxel_groups4[i])
        elif [unique_contacts4[i][1], unique_contacts4[i][0], unique_contacts4[i][3], unique_contacts4[i][2]] in neighborr:
            unique_contacts5.append(unique_contacts4[i])
            voxel_groups5.append(voxel_groups4[i])
        else:
            pass

    vca_weights = np.array(vca_weights)
    atom_voxels = []
    #groups = []
    #group_weights = []
    #group_scales = []
    coeffs = []
    bptable = []

    cubic_grid_big = np.array(cubic_grid_big)
    parts = []
    for i in range(len(structure)):
        atom_voxels.append(0)
        coeffs.append([0 for j in range(49)])
        if i in filter_indicies: # is this atom in the substructure of interest?
            for j in range(len(predictions_rfr)):
                if i in used_for[j]:
                    neigh = neighborr[j]
                    try:
                        index = unique_contacts5.index(neigh)
                    except:
                        try:
                            index = unique_contacts5.index([neigh[1], neigh[0], neigh[3], neigh[2]])
                        except:
                            break
                    vg = voxel_groups5[index]
                    points = np.take(cubic_grid_big, vg, axis=0) - structure[i].coords
                    weights = np.take(vca_weights, vg)

                    if i == neigh[0]:
                        s0 = structure[neigh[0]]
                        s1 = structure[neigh[1]]
                    else:
                        s0 = structure[neigh[1]]
                        s1 = structure[neigh[0]]
                    # While it seems unintuitive to multiply the weighted pressure prediction by sum(weights)/len(weights),
                    # this is entirely necessary for the bptable values to be correct.
                    prediction = sum(weights)*neighbor_weights[j]*predictions_rfr[j]/len(vg)
                    bptable.append([s0.specie.symbol, s0.coords, s1.specie.symbol, s1.coords, prediction, sum(weights), 0.5*sum(weights)*structure.volume/vmt])

                    """
                    # This code is used for diagnostics. Don't delete it, but
                    # don't run it unless necessary either, it slows the main
                    # for loop down by about a factor of 3.
                    l1 = list(neighborr[j][:2])
                    l2 = list(np.flip(neighborr[j][:2]))
                    if l1 in groups:
                        group_weights[groups.index(l1)].append(np.sum(weights))
                        group_scales[groups.index(l1)].append(predictions_rfr[j])
                    elif l2 in groups:
                        group_weights[groups.index(l2)].append(np.sum(weights))
                        group_scales[groups.index(l2)].append(predictions_rfr[j])
                    else:
                        groups.append(l1)
                        group_weights.append([np.sum(weights)])
                        group_scales.append([predictions_rfr[j]])
                    """

                    atom_voxels[-1] += sum(weights)

                    scale = weights*neighbor_weights[j]*predictions_rfr[j]/len(vg)
                    h = np.hypot(np.hypot(points[:,0], points[:,1]), points[:,2])
                    phi = np.arctan2(points[:,1], points[:,0])
                    theta = np.arccos(points[:,2]/h)
                    cost = np.cos(theta)
                    sint = np.sin(theta)
                    cosp = np.cos(phi)
                    sinp = np.sin(phi)


                    coeffs[i][0] += np.sum(np.ones(len(points))*scale*0.5*(1/math.pi)**0.5)

                    coeffs[i][1] += np.sum(scale*0.5*(3/(math.pi))**0.5*cost)
                    coeffs[i][2] += np.sum(-scale*(3/(8*math.pi))**0.5*sint*cosp*(2)**0.5)
                    coeffs[i][3] += np.sum(-scale*(3/(8*math.pi))**0.5*sint*sinp*(2)**0.5)

                    coeffs[i][4] += np.sum(scale*0.25*(5/(math.pi))**0.5*(3*cost**2-1))
                    coeffs[i][5] += np.sum(-scale*0.5*(15/(math.pi))**0.5*sint*cosp*cost)
                    coeffs[i][6] += np.sum(-scale*0.5*(15/(math.pi))**0.5*sint*sinp*cost)
                    coeffs[i][7] += np.sum(scale*0.25*(15/(math.pi))**0.5*sint**2*np.cos(2*phi))
                    coeffs[i][8] += np.sum(scale*0.25*(15/(math.pi))**0.5*sint**2*np.sin(2*phi))

                    coeffs[i][9] += np.sum(scale*0.25*(7/(math.pi))**0.5*(5*cost**3-3*cost))
                    coeffs[i][10] += np.sum(-scale*0.125*(21/(math.pi))**0.5*sint*(5*cost**2-1.0)*cosp*(2)**0.5)
                    coeffs[i][11] += np.sum(-scale*0.125*(21/(math.pi))**0.5*sint*(5*cost**2-1.0)*sinp*(2)**0.5)
                    coeffs[i][12] += np.sum(scale*0.25*(105/(2*math.pi))**0.5*sint**2*(cost)*np.cos(2*phi)*(2)**0.5)
                    coeffs[i][13] += np.sum(scale*0.25*(105/(2*math.pi))**0.5*sint**2*(cost)*np.sin(2*phi)*(2)**0.5)
                    coeffs[i][14] += np.sum(-scale*0.125*(35/(math.pi))**0.5*sint**3*np.cos(3*phi)*(2)**0.5)
                    coeffs[i][15] += np.sum(-scale*0.125*(35/(math.pi))**0.5*sint**3*np.sin(3*phi)*(2)**0.5)

                    coeffs[i][16] += np.sum(scale*(3/16)*(1/(math.pi))**0.5*(35*cost**4-30*cost**2+3))
                    coeffs[i][17] += np.sum(-scale*(3/8)*(5/(math.pi))**0.5*sint*(7*cost**3-3*cost)*cosp*(2)**0.5)
                    coeffs[i][18] += np.sum(-scale*(3/8)*(5/(math.pi))**0.5*sint*(7*cost**3-3*cost)*sinp*(2)**0.5)
                    coeffs[i][19] += np.sum(scale*(3/8)*(5/(2*math.pi))**0.5*sint**2*(7*cost**2-1)*np.cos(2*phi)*(2)**0.5)
                    coeffs[i][20] += np.sum(scale*(3/8)*(5/(2*math.pi))**0.5*sint**2*(7*cost**2-1)*np.sin(2*phi)*(2)**0.5)
                    coeffs[i][21] += np.sum(-scale*(3/8)*(35/(math.pi))**0.5*sint**3*cost*np.cos(3*phi)*(2)**0.5)
                    coeffs[i][22] += np.sum(-scale*(3/8)*(35/(math.pi))**0.5*sint**3*cost*np.sin(3*phi)*(2)**0.5)
                    coeffs[i][23] += np.sum(scale*(3/16)*(35/(2*math.pi))**0.5*sint**4*np.cos(4*phi)*(2)**0.5)
                    coeffs[i][24] += np.sum(scale*(3/16)*(35/(2*math.pi))**0.5*sint**4*np.sin(4*phi)*(2)**0.5)
    data = []
    net_pressure = 0

    for i in range(len(atom_voxels)):
        net_pressure += atom_voxels[i] * 29421.0265*coeffs[i][0]*np.sqrt(4*math.pi) / 2
        data.append([structure[i].specie.symbol, '{0:.2f} GPa'.format(round(29421.0265*coeffs[i][0]*np.sqrt(4*math.pi),2))])
    if temp == None:
        data.append(["Total", '{0:.2f} GPa'.format(round(net_pressure / sum(atom_voxels),2))])
    else:
        data.append(["Total", "undeterminable"])

    sysname = cif.split(".")[0]
    qps = []
    qp_lmax = []
    qps_total = []
    qp_lines = ["_"*85+"\n", "  CP Quadrupole Report\n"]
    for i in range(len(coeffs)):
        qps.append([0,0,0,0,0])
        for j in range(5):
            for k in range(j**2,(j+1)**2):
                qps[i][j] += coeffs[i][k]**2
    for i in range(len(qps)):
        qp_lmax.append([0,0,0])
        qps_total.append([])
        for j in range(3):
            for k in range(j+3):
                qp_lmax[i][j] += qps[i][k]
        for j in range(3):
            qps_total[i].append(qps[i][2]/qp_lmax[i][j])
    for i in range(len(qps_total)):
        qp_lines.append("    Atom {0:3n} ({1:2s})    (lmax = 2): {2:.6f}  (lmax = 3): {3:.6f}  (lmax = 4): {4:.6f}\n".format(i+1, geo[i][0], qps_total[i][0], qps_total[i][1], qps_total[i][2]))
    qp_lines.append("_"*85+"\n")

    pm = ["0p", "1p", "1m", "2p", "2m", "3p", "3m", "4p", "4m", "5p", "5m", "6p", "6m"]
    coeff_labels = ["l_"+str(i)+"m_"+pm[j]+"=" for i in range(7) for j in range(2*i+1)]
    coeff_labels[0] = coeff_labels[0].replace("0p=", "0=")
    pcell = "\n".join(["    " + "    ".join(["{:.6f}".format(j) for j in i]) for i in cell])
    pgeo = "\n".join(["      ".join([i[0]]+["{:.6f}".format(j) for j in i[1:]]) for i in geo])
    pcoeffs = "\n".join(["\n".join([coeff_labels[j]+"      {:.14f}".format(i[j]) for j in range(len(i))]) for i in coeffs])
    modifile(sysname+"_MLCP-cell", "w", pcell)
    modifile(sysname+"_MLCP-geo", "w", pgeo)
    modifile(sysname+"_MLCP-coeff", "w", pcoeffs)

    t7 = time.time()

    """
    print(round(t2-t1,3))
    print(round(t3-t2,3))
    print(round(t4-t3,3))
    print(round(t5-t4,3))
    print(round(t6-t5,3))
    print(round(t7-t6,3))
    print(round(t7-t1,3))
    """

    f = open(sysname+"_MLCP-data","w")
    f.write("Based on: "+cif+"\nRan at Scale Factor: "+str(sf)+"\n")
    print("\n")
    for i in summary:
        print("   ",i)
        f.write(i+"\n")
    for i in qp_lines:
        f.write(i)
    for i in data:
        print("   ", i[0], ": ", i[1])
        f.write(i[0]+": "+i[1]+"\n")
    print("    Total time:", round(t7-t1, 2), "sec.")
    f.write("Total time: "+str(round(t7-t1, 2))+" sec.\n")
    f.close()
    print("\n")

    bptable_lines = []
    for i in range(len(bptable)):
        bptable_lines.append(bptable[i][0]+" ")
        for j in range(len(bptable[i][1])):
            bptable_lines[i] += '{0:.6f} '.format(bptable[i][1][j])
        bptable_lines[i] += "to " + bptable[i][2] + " "
        for j in range(len(bptable[i][3])):
            bptable_lines[i] += '{0:.6f} '.format(bptable[i][3][j])
        bptable_lines[i] += "dist = {0:.6f}".format(np.linalg.norm(bptable[i][1]-bptable[i][3]))
        bptable_lines[i] += " pressure = {0:.6f} {1:.6f} {2:.6f}\n".format(bptable[i][4], bptable[i][5], bptable[i][6])
    modifile(sysname+"_MLCP-bptable","w",bptable_lines)

    #return(to_js([cell, geo, coeffs, template3, data]))

import sys
if len(sys.argv) == 1:
    print("Usage: python3 mlcp.py <cif name> <scale factor (optional)> <xyz template name(optional)>\n")
    exit()
elif len(sys.argv) == 2:
    print_cif(sys.argv[1])
elif len(sys.argv) == 3:
    print_cif(sys.argv[1], sys.argv[2])
elif len(sys.argv) == 4:
    print_cif(sys.argv[1], sys.argv[2], sys.argv[3])
else:
    print("Usage: python3 mlcp.py <cif name> <scale factor (optional)> <xyz template name(optional)>\n")
    exit()

