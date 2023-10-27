"""
mcmd.py
A package to perform simulations of guest-loaded MOFs in the osmotic ensemble, based on the yaff MD engine

Handles the primary functions
"""

import os, sys
import numpy as np
np.random.seed(42)
from copy import deepcopy
from molmod.constants import boltzmann, avogadro
from molmod.constants import planck as h
from molmod.units import angstrom, kjmol, kelvin, femtosecond, bar, kcalmol, kilogram
from molmod import Molecule
from time import time
from scipy.spatial import cKDTree
import h5py as h5
from copy import copy, deepcopy

from osmotic_mcmd.utilities import Acceptance, Parse_data, random_ads, random_rot
from wrapper_ewald import Sfac, ewald_insertion, ewald_deletion, ewald_displace, ewald_from_sfac
from wrapper_forceparts import electrostatics, electrostatics_realspace, electrostatics_realspace_insert, MM3, MM3_insert, LJ, LJ_insert

from yaff.external.lammps_generator import *
from yaff.external.lammpsio import *
from yaff.external.liblammps import *
from yaff import log
from yaff.pes.colvar import CVVolume
from yaff.sampling.enhanced import *

from yaff import System, ForceField, XYZWriter, VerletScreenLog, MTKBarostat, \
           NHCThermostat, TBCombination, VerletIntegrator, HDF5Writer, log

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


if rank==0:
    log.set_level(log.medium)
else:
    log.set_level(log.silent)



class MCMD():
    def __init__(self, system_file, adsorbate_file, ff_file, T, P, fugacity, MD_trial_fraction, rcut, fixed_N = None, write_h5s = False, barostat = True, vol_constraint = False, write_traj = False, meta = False, timestep = 0.5*femtosecond, MDsteps = 600):

        self.ff_file = ff_file
        self.T = T
        self.beta = 1/(boltzmann*T)
        self.P = P
        self.fugacity = fugacity
        self.prob = np.array([0.5, 0.5, MD_trial_fraction], dtype=float)
        self.prob = np.cumsum(self.prob)/sum(self.prob)
        self.rcut = rcut

        data = Parse_data(system_file, adsorbate_file, ff_file, self.rcut)
        self.data = data
        if np.max(np.abs(data.charges_ads)) < 1e-8:
            self.ads_ei = False
        else:
            self.ads_ei = True
        assert data.ei == True
        assert data.mm3 == True or data.lj == True

        self.pos = data.pos_MOF
        self.rvecs = data.rvecs
        self.V = np.linalg.det(self.rvecs)
        self.rvecs_flat = self.rvecs.reshape(9)
        self.N_frame = len(self.pos)
        self.nads = len(data.pos_ads)
        self.charges = data.charges_MOF
        self.pos_ads = data.pos_ads
        self.n_ad = len(self.pos_ads)
        self.Z_ads = 0
        self.fixed_N = fixed_N
        self.count_mds = 0
        self.write_h5s = write_h5s
        self.barostat = barostat
        self.vol_constraint = vol_constraint

        self.alpha_scale = 3.2
        self.gcut_scale = 1.0
        self.alpha = self.alpha_scale / self.rcut
        self.gcut = self.gcut_scale * self.alpha
        self.step = 1.0 * angstrom

        if self.ads_ei:
            self.sfac = Sfac(self.pos, self.N_frame, self.rvecs_flat, self.charges, self.alpha, self.gcut)
        self.e_el_real = 0
        self.e_vdw = 0

        self.write_traj = write_traj
        self.meta = meta
        self.timestep = timestep
        self.MDsteps = MDsteps

        if rank == 0:
            if self.fixed_N:
                if os.path.exists('results/output_%d.h5'%self.fixed_N):
                    os.remove('results/output_%d.h5'%self.fixed_N)
                if os.path.exists('results/temp_%d.h5'%self.fixed_N):
                    os.remove('results/temp_%d.h5'%self.fixed_N)
            else:
                if os.path.exists('results/output_%.8f.h5'%(self.P/bar)):
                    os.remove('results/output_%.8f.h5'%(self.P/bar))
                if os.path.exists('results/temp_%.8f.h5'%(self.P/bar)):
                    os.remove('results/temp_%.8f.h5'%(self.P/bar))



    def overlap(self, pos):
        tree = cKDTree(pos, compact_nodes=False, copy_data=False, balanced_tree=False)
        pairs = tree.query_pairs(0.05*angstrom)
        return len(list(pairs)) > 0


    def compute_insertion(self, new_pos):
        n = self.Z_ads
        plen = len(self.pos)

        if self.ads_ei:
            self.sfac = np.array(ewald_insertion(self.sfac, new_pos, self.rvecs_flat, self.data.charges_ads, self.alpha, self.gcut))
            e_ewald = ewald_from_sfac(self.sfac, self.rvecs_flat, self.alpha, self.gcut)
            self.e_el_real += electrostatics_realspace_insert(self.N_frame, len(self.pos)-self.nads, self.pos, self.rvecs_flat, self.data.charges[:plen], self.data.radii[:plen], self.rcut, self.alpha, self.gcut)
        else:
            e_ewald = 0
            self.e_el_real = 0

        if(self.data.mm3):
            self.e_vdw += MM3_insert(self.pos, len(self.pos)-self.nads, self.N_frame, self.rvecs_flat, self.data.sigmas[:plen], self.data.epsilons[:plen], self.rcut)
        elif(self.data.lj):
            self.e_vdw += LJ_insert(self.pos, len(self.pos)-self.nads, self.N_frame, self.rvecs_flat, self.data.sigmas[:plen], self.data.epsilons[:plen], self.rcut)

        return e_ewald + self.e_el_real + self.e_vdw


    def compute_deletion(self, deleted_coord):
        n = self.Z_ads
        new_pos = np.append(self.pos, deleted_coord, axis=0)
        plen = len(new_pos)

        if self.ads_ei:
            self.sfac = np.array(ewald_deletion(self.sfac, deleted_coord, self.rvecs_flat, self.data.charges_ads, self.alpha, self.gcut))
            e_ewald = ewald_from_sfac(self.sfac, self.rvecs_flat, self.alpha, self.gcut);
            self.e_el_real -= electrostatics_realspace_insert(self.N_frame, len(self.pos), new_pos, self.rvecs_flat, self.data.charges[:plen], self.data.radii[:plen], self.rcut, self.alpha, self.gcut)
        else:
            e_ewald = 0
            self.e_el_real = 0

        if(self.data.mm3):
            self.e_vdw -= MM3_insert(new_pos, len(self.pos), self.N_frame, self.rvecs_flat, self.data.sigmas[:plen], self.data.epsilons[:plen], self.rcut)
        elif(self.data.lj):
            self.e_vdw -= LJ_insert(new_pos, len(self.pos), self.N_frame, self.rvecs_flat, self.data.sigmas[:plen], self.data.epsilons[:plen], self.rcut)

        return e_ewald + self.e_el_real + self.e_vdw


    def compute(self):
        n = self.Z_ads
        plen = len(self.pos)

        t = time()
        if(n == 0):
            return 0

        e_el = electrostatics(self.pos, self.N_frame, self.Z_ads, self.rvecs_flat, self.data.charges[:plen], self.data.radii[:plen], self.rcut, self.alpha, self.gcut)

        if(self.data.mm3):
            e_vdw = MM3(self.pos, self.N_frame, self.Z_ads, self.rvecs_flat, self.data.sigmas[:plen], self.data.epsilons[:plen], self.rcut)
        elif(self.data.lj):
            e_vdw = LJ(self.pos, self.N_frame, self.Z_ads, self.rvecs_flat, self.data.sigmas[:plen], self.data.epsilons[:plen], self.rcut)

        return e_el + e_vdw


    def insertion(self, new_pos):
        self.Z_ads += 1

        self.pos = np.append(self.pos, new_pos, axis=0)
        e_new = self.compute_insertion(new_pos)

        return e_new


    def deletion(self):
        iatom = np.random.randint(self.Z_ads)
        index = self.N_frame + self.nads*iatom
        self.Z_ads -= 1

        deleted_coord = deepcopy(self.pos[index:index+self.nads])
        self.pos = np.delete(deepcopy(self.pos), np.s_[index:index+self.nads], axis=0)
        e_new = self.compute_deletion(deleted_coord)

        return deleted_coord, e_new


    def write_traj_func(self, traj, symbols):
        if rank == 0:
            f = open('results/fixed_N_trajectory_%d.xyz'%self.fixed_N, 'w')
            for iframe, frame in enumerate(traj):
                f.write('%d\nsnapshot %d\n'%(len(frame), iframe))
                for el, pos in zip(symbols, frame):
                    f.write('%s %f %f %f\n'%(el, pos[0]/angstrom, pos[1]/angstrom, pos[2]/angstrom))
            f.close()


    def append_h5(self, iteration):

        if rank == 0:
            datasets = {'cell':[], 'cons_err':[], 'econs':[], 'pos':[], 'press':[], 'ptens':[], 'temp':[], 'volume':[]}

            if self.fixed_N:
                temp = 'results/temp_%d.h5'%self.fixed_N
                previous = 'results/output_%d.h5'%self.fixed_N
            else:
                temp = 'results/temp_%.8f_%i.h5'%(self.P/bar, iteration)
                previous = 'results/output_%.8f.h5'%(self.P/bar)

            if os.path.exists(previous):
                f_prev = h5.File(previous, 'r')
                for key, _ in datasets.items():
                    datasets[key] = f_prev['trajectory/%s'%key][:]
                f_prev.close()
                datasets['pos'] = datasets['pos'][:, :self.N_frame, :]

            f_tmp = h5.File(temp, 'r')
            for key, _ in datasets.items():
                if len(datasets[key]) == 0:
                    datasets[key] = f_tmp['trajectory/%s'%key][:]
                    if key == 'pos':
                        datasets[key] = datasets[key][:, :self.N_frame, :]
                else:
                    if key == 'pos':
                        datasets[key] = np.concatenate((datasets[key], f_tmp['trajectory/%s'%key][:, :self.N_frame, :]), axis=0)
                    else:
                        datasets[key] = np.concatenate((datasets[key], f_tmp['trajectory/%s'%key][:]), axis=0)

            f_tmp.close()

            os.remove(temp)
            if os.path.exists(previous):
                os.remove(previous)

            f = h5.File(previous, 'a')
            f.create_group('trajectory')
            for key, value in datasets.items():
                f.create_dataset(name = 'trajectory/%s'%key, data = value)
            f.close()


    def run_GCMC(self, N_iterations, N_sample):

        A = Acceptance()

        if rank == 0:
            if not (os.path.isdir('results')):
                try:
                    os.mkdir('results')
                except:pass

            if self.write_traj:
                ftraj = open('results/traj_%.8f.xyz'%(self.P/bar), 'w')

        e = 0
        t_it = time()

        N_samples = []
        E_samples = []
        pressures = []
        traj = []
        q0s = []

        if rank == 0:
            print('\n Iteration  inst. N    inst. E    inst. V     time [s]')
            print('--------------------------------------------------------')

        for iteration in range(N_iterations+1):

            if self.ads_ei:
                sfac_init = deepcopy(self.sfac)
            pos_init = deepcopy(self.pos)
            rvecs_init = deepcopy(self.rvecs)
            rvecs_flat_init = deepcopy(self.rvecs_flat)
            V_init = self.V
            e_el_real_init = self.e_el_real
            e_vdw_init = self.e_vdw
            switch = np.random.rand()
            acc = 0

            # Insertion / deletion
            if(switch < self.prob[0] and not self.Z_ads == self.fixed_N):

                if(switch < self.prob[0]/2):

                    new_pos = random_ads(self.pos_ads, self.rvecs)
                    e_new = self.insertion(new_pos)

                    exp_value = self.beta * (-e_new + e)
                    if(exp_value > 100):
                        acc = 1
                    elif(exp_value < -100):
                        acc = 0
                    else:
                        acc = min(1, self.V*self.beta*self.fugacity/self.Z_ads * np.exp(exp_value))

                    # Reject monte carlo move
                    if np.random.rand() > acc:
                        self.pos = pos_init
                        if self.ads_ei:
                            self.sfac = sfac_init
                        self.e_el_real = e_el_real_init
                        self.e_vdw = e_vdw_init
                        self.Z_ads -= 1
                    else:
                        e = e_new

                elif(self.Z_ads > 0):

                    deleted_coord, e_new = self.deletion()

                    exp_value = -self.beta * (e_new - e)
                    if(exp_value > 100):
                        acc = 1
                    else:
                        acc = min(1, (self.Z_ads+1)/self.V/self.beta/self.fugacity * np.exp(exp_value))

                    # Reject monte carlo move
                    if np.random.rand() > acc:
                        self.pos = pos_init
                        if self.ads_ei:
                            self.sfac = sfac_init
                        self.e_el_real = e_el_real_init
                        self.e_vdw = e_vdw_init
                        self.Z_ads += 1
                    else:
                        e = e_new

            elif(switch < self.prob[1]):

                if self.Z_ads != 0:

                    trial = np.random.randint(self.Z_ads)

                    if((switch < self.prob[0] + (self.prob[1]-self.prob[0])/2) or self.nads == 1):

                        # Calculate translation energy as deletion + insertion of molecule
                        deleted_coord, e_new = self.deletion()
                        deleted_coord += self.step * (np.random.rand(3) - 0.5)
                        e_new = self.insertion(deleted_coord)

                    else:

                        # Calculate rotation energy as deletion + insertion of molecule
                        deleted_coord, e_new = self.deletion()
                        deleted_coord = random_rot(deleted_coord, circlefrac=0.1)
                        e_new = self.insertion(deleted_coord)

                    exp_value = -self.beta * (e_new - e)
                    if(exp_value > 0):
                        exp_value = 0
                    acc = min(1, np.exp(exp_value))

                    # Reject monte carlo move
                    if np.random.rand() > acc:
                        self.pos = pos_init
                        if self.ads_ei:
                            self.sfac = sfac_init
                        self.e_el_real = e_el_real_init
                        self.e_vdw = e_vdw_init
                    else:
                        e = e_new

            else:

                # Construct system and forcefield class for the MD engine
                from yaff import System, ForceField, XYZWriter, VerletScreenLog, MTKBarostat, \
                       NHCThermostat, TBCombination, VerletIntegrator, HDF5Writer, log
                log.set_level(0)

                n = np.append(self.data.numbers_MOF, np.tile(self.data.numbers_ads, self.Z_ads))

                ffa_MOF = self.data.system.ffatypes[self.data.system.ffatype_ids]
                ffa_ads = self.data.system_ads.ffatypes[self.data.system_ads.ffatype_ids]
                ffa = np.append(ffa_MOF, np.tile(ffa_ads, self.Z_ads))
                assert len(self.pos) == len(ffa)

                s = System(n, self.pos, ffatypes = ffa, rvecs=self.rvecs)
                s.detect_bonds()

                ff = ForceField.generate(s, self.ff_file,
                                            rcut=self.rcut,
                                            alpha_scale=self.alpha_scale,
                                            gcut_scale=self.gcut_scale,
                                            tailcorrections=True)

                ff_lammps = swap_noncovalent_lammps(ff, fn_system='system_%.8f.dat'%(self.P/bar),
                                        fn_table='table.dat',
                                        nrows=5000,
                                        kspace='pppm',
                                        kspace_accuracy=1e-7,
                                        scalings_ei = [1.0, 1.0, 1.0],
                                        move_central_cell=False,
                                        fn_log="none",
                                        overwrite_table=False, comm=comm)

                # Setup and NPT MD run
                if rank == 0:
                    vsl = VerletScreenLog(step=50)

                    if self.write_h5s:
                        if self.fixed_N:
                            h5file = h5.File('results/temp_%d.h5'%self.fixed_N, mode='w')
                            hdf5_writer = HDF5Writer(h5file, step=101)
                        else:
                            h5file = h5.File('results/temp_%.8f_%i.h5'%(self.P/bar, iteration), mode='w')
                            hdf5_writer = HDF5Writer(h5file, step=101)

       	       	ensemble_hook = NHCThermostat(temp=self.T, timecon=100*femtosecond, chainlength=3)
                if self.barostat:
                    mtk = MTKBarostat(ff_lammps, temp=self.T, press=self.P, \
                        timecon=1000*femtosecond, vol_constraint = self.vol_constraint, anisotropic = True)
                    ensemble_hook = TBCombination(ensemble_hook, mtk)

                if self.meta:
                    cv = CVVolume(ff_lammps.system)
                    sigma = 1000*angstrom**3
                    K = 20*kjmol
                    step = 498

                # Run MD
                t = time()
                if self.write_h5s:
                    if rank == 0:
                        verlet = VerletIntegrator(ff_lammps, self.timestep, hooks=[ensemble_hook, vsl, hdf5_writer], temp0=self.T)
                    else:
                        verlet = VerletIntegrator(ff_lammps, self.timestep, hooks=[ensemble_hook], temp0=self.T)
                else:
                    if rank == 0:
                        hooks = [ensemble_hook, vsl]
                        if self.meta:
                            meta = MTDHook(ff_lammps, cv, sigma, K, start=step, step=step)
                            for q0 in q0s:
                                meta.hills.add_hill(q0, K)
                            hooks.append(meta)
                        verlet = VerletIntegrator(ff_lammps, self.timestep, hooks=hooks, temp0=self.T)
                    else:
                        hooks = [ensemble_hook]
                        if self.meta:
                            meta = MTDHook(ff_lammps, cv, sigma, K, start=step, step=step) 
                            for q0 in q0s:
                                meta.hills.add_hill(q0, K)
                            hooks.append(meta)
                        verlet = VerletIntegrator(ff_lammps, self.timestep, hooks=hooks, temp0=self.T)

                e0_tot = verlet._compute_ekin() + ff_lammps.compute()
                verlet.run(self.MDsteps)
                ef_tot = verlet._compute_ekin() + ff_lammps.compute()

                if not self.vol_constraint:
                    Vn = np.linalg.det(ff_lammps.system.cell.rvecs)
                    exp_value = -self.beta * (ef_tot - e0_tot + self.P * (Vn - self.V) - len(self.pos)/self.beta * np.log(Vn/self.V))
                else:
                    exp_value = -self.beta * (ef_tot - e0_tot)

                if(exp_value > 0):
                    exp_value = 0
                acc = min(1, np.exp(exp_value))

                if self.write_h5s:
                    if rank == 0:
                        h5file.close()

                # Accept monte carlo move
                if np.random.rand() < acc:
                    print('MD accepted')
                    if self.write_h5s:
                        # Append MD data to previous data
                        self.append_h5(iteration)

                    # Rebuild data for MC
                    pos_total = ff_lammps.system.pos
                    self.pos = pos_total[:self.N_frame]
                    pos_molecules = pos_total[self.N_frame:]

                    self.rvecs = ff_lammps.system.cell.rvecs
                    self.rvecs_flat = self.rvecs.reshape(9)
                    self.V = np.linalg.det(self.rvecs)
                    if self.meta:
                        q0s.append(self.V)
                    if self.ads_ei:
                        self.sfac = Sfac(self.pos, self.N_frame, self.rvecs_flat, \
                                            self.charges, self.alpha, self.gcut)
                    self.e_el_real = 0
                    self.e_vdw = 0

                    if self.Z_ads > 0:
                        for p in np.split(pos_molecules, self.Z_ads):
                            e_new = self.insertion(p)
                            self.Z_ads -= 1
                        e = e_new
                    else:
                        e = 0
                else:
                    print('MD not accepted')
                    self.pos = pos_init
                    self.rvecs = rvecs_init
                    self.rvecs_flat = rvecs_flat_init
                    self.V = V_init

                if rank == 0:
                    log.set_level(log.medium)

            if(iteration % N_sample == 0 and iteration > 0):
                eprint = e
                if np.abs(eprint) < 1e-10:
                    eprint = 0
                if rank == 0:
                    print(' {:7.7}       {:7.7} {:7.7} {:7.7}    {:7.4}'.format(
                          str(iteration),str(self.Z_ads),str(eprint/kjmol),str(self.V/angstrom**3),time()-t_it)
                          )
                t_it = time()
                N_samples.append(self.Z_ads)
                E_samples.append(e)
                if self.Z_ads == self.fixed_N:
                    traj.append(self.pos)

                if rank == 0 and self.write_traj:

                    natom = self.N_frame + self.nads * self.Z_ads
                    rv = self.rvecs_flat/angstrom
                    ffa_MOF = self.data.system.ffatypes[self.data.system.ffatype_ids]
                    ffa_ads = self.data.system_ads.ffatypes[self.data.system_ads.ffatype_ids]
                    ffa = np.append(ffa_MOF, np.tile(ffa_ads, self.Z_ads))

                    ftraj.write('%d\n%f %f %f %f %f %f %f %f %f\n'%(natom, rv[0], rv[1], rv[2], rv[3], rv[4], rv[5], rv[6], rv[7], rv[8]))
                    for s, p in zip(ffa, self.pos/angstrom):
                        ftraj.write('%s %f %f %f\n'%(s, p[0], p[1], p[2]))


        if rank == 0:
            print('Average N: %.3f'%np.average(N_samples))
            if self.fixed_N:
                np.save('results/N_%d.npy'%self.fixed_N, np.array(N_samples))
                np.save('results/E_%d.npy'%self.fixed_N, np.array(E_samples))
            else:
                np.save('results/N_%.8f.npy'%(self.P/bar), np.array(N_samples))
                np.save('results/E_%.8f.npy'%(self.P/bar), np.array(E_samples))

            if self.fixed_N:

                from yaff import System
                n = np.append(self.data.numbers_MOF, np.tile(self.data.numbers_ads, self.Z_ads))
                s = System(n, self.pos, rvecs=self.rvecs)
                s.to_file('results/end_%d.xyz'%self.fixed_N)

                mol = Molecule.from_file('results/end_%d.xyz'%self.fixed_N)
                symbols = mol.symbols
                self.write_traj_func(traj, symbols)
                os.remove('results/end_%d.xyz'%self.fixed_N)

            if self.write_traj:
                ftraj.close()


class Widom():
    def __init__(self, system_file, adsorbate_file, ff_file, T, rcut, write_all = False):

        self.ff_file = ff_file
        self.T = T
        self.beta = 1/(boltzmann*T)
        self.rcut = rcut

        data = Parse_data(system_file, adsorbate_file, ff_file, self.rcut)
        self.data = data

        assert data.mm3 == True or data.lj == True

        self.pos = data.pos_MOF
        self.rvecs = data.rvecs
        self.V = np.linalg.det(self.rvecs)
        self.rvecs_flat = self.rvecs.reshape(9)
        self.N_frame = len(self.pos)
        self.nads = len(data.pos_ads)
        self.charges = data.charges_MOF
        self.pos_ads = data.pos_ads
        self.n_ad = len(self.pos_ads)
        self.mass = data.mass_MOF

        self.alpha_scale = 3.2
        self.gcut_scale = 1.0
        self.alpha = self.alpha_scale / self.rcut
        self.gcut = self.gcut_scale * self.alpha

        self.sfac = Sfac(self.pos, self.N_frame, self.rvecs_flat, self.charges, self.alpha, self.gcut)
        self.sfac_frame = deepcopy(self.sfac)
        self.e_el_real, self.e_vdw = 0, 0
        self.write_all = write_all


    def compute_insertion(self, new_pos):
        n = 1
        plen = len(self.pos)

        self.sfac = np.array(ewald_insertion(self.sfac, new_pos, self.rvecs_flat, self.data.charges_ads, self.alpha, self.gcut))
        e_ewald = ewald_from_sfac(self.sfac, self.rvecs_flat, self.alpha, self.gcut)
        self.e_el_real += electrostatics_realspace_insert(self.N_frame, len(self.pos)-self.nads, self.pos, self.rvecs_flat, self.data.charges[:plen], self.data.radii[:plen], self.rcut, self.alpha, self.gcut)

        if(self.data.mm3):
            self.e_vdw += MM3_insert(self.pos, len(self.pos)-self.nads, self.N_frame, self.rvecs_flat, self.data.sigmas[:plen], self.data.epsilons[:plen], self.rcut)
        elif(self.data.lj):
            self.e_vdw += LJ_insert(self.pos, len(self.pos)-self.nads, self.N_frame, self.rvecs_flat, self.data.sigmas[:plen], self.data.epsilons[:plen], self.rcut)

        return e_ewald + self.e_el_real + self.e_vdw

    def run_widom(self, N_iterations, N_sample):

        if not (os.path.isdir('results')):
            os.mkdir('results')

        E_samples = []

        print('\n Iteration  inst. Hads [kJ/mol]  inst. K_H [mol/kg/bar]  time [s]')
        print('-------------------------------------------------------------------')

        t_it = time()

        for iteration in range(N_iterations):

            new_pos = random_ads(self.pos_ads, self.rvecs)
            self.pos = np.append(self.pos, new_pos, axis=0)
            e_insertion = self.compute_insertion(new_pos)
            E_samples.append(e_insertion)

            # Reset e_el_real, e_vdw, sfac and pos
            self.sfac = deepcopy(self.sfac_frame)
            self.e_el_real, self.e_vdw = 0, 0
            self.pos = self.pos[:-self.n_ad]

            if iteration > 0 and iteration % N_sample == 0:

                E_temp = np.array(E_samples)

                # if adsorption energies are very positive, a division by 0 error will occur here:
                try:
                    E_ads = np.average(E_temp*np.exp(-self.beta*E_temp))/np.average(np.exp(-self.beta*E_temp))
                except:
                    E_m = min(E_temp)
                    E_ads = np.average(E_temp*np.exp(-self.beta*(E_temp-E_m)))/np.average(np.exp(-self.beta*(E_temp-E_m)))

                rho = self.mass/np.linalg.det(self.rvecs)
                K_H = self.beta/rho*np.average(np.exp(-self.beta*E_temp))

                print(' {:7.7}       {:7.7}                {:7.7}         {:7.4}'.format(
                      str(iteration),str((E_ads - 1/self.beta)/kjmol),str(K_H/(avogadro/(kilogram*bar))), time()-t_it)
                      )
                t_it = time()

        E_samples = np.array(E_samples)

        if self.write_all:
            np.save('results/Widom_E.npy', np.array(E_samples))
        else:
            rho = self.mass/np.linalg.det(self.rvecs)
            # Do bootstrapping
            def bootstrap(data, type='Hads'):
                means = []
                for i in range(100):
                    means.append(sample_mean(data, type))
                return np.average(means), np.std(np.array(means))

            def sample_mean(data, type):
                resampled = np.random.choice(data, len(data), replace=True)
                if type == 'Hads':
                    try:
                        return (np.average(resampled*np.exp(-self.beta*resampled))/np.average(np.exp(-self.beta*resampled)) - 1/self.beta)/kjmol
                    except:
                        E_m = min(resampled)
                        return (np.average(resampled*np.exp(-self.beta*(resampled-E_m)))/np.average(np.exp(-self.beta*(resampled-E_m))) - 1/self.beta)/kjmol
                elif type == 'KH':
                    return np.average(self.beta/rho*np.exp(-self.beta*resampled) / (avogadro/(kilogram*bar)))

            H_ads_av, H_ads_std = bootstrap(E_samples, type='Hads')
            K_H_av, K_H_std = bootstrap(E_samples, type='KH')

            bootstrap_data = np.array([[H_ads_av, H_ads_std], [K_H_av, K_H_std]])
            print(bootstrap_data)

            np.save('results/Widom_result.npy', bootstrap_data)

