#Module traj_convert
#Created by Aria Coraor

import argparse
import numpy as np
import mdtraj as md
import math
from vect_quat_util import *
import os
import matplotlib.pyplot as plt

def main():
	"""For a given pair of sequence files, calculate the ideal instantaneous
	FRET for the frozen positions.
	"""

	#Load curved backbone pdb
	dna_xyz, dna_top = load_dna(args.base)

	#Load sequence files
	cy3_seq, cy5_seq = load_seqs(args.c3s,args.c5s)

	#Calculate prebuilt Cy3 structural params
	cy3_params = calc_cy3_params(args.preb)	

	#Calculate cy3, cy5 geometry
	geom = calc_geom(dna_xyz,dna_top,cy3_seq,cy5_seq,cy3_params)
	geom['r0'] = 6.5

	#Calculate k2, FRET for rigid system
	rig_k2,rig_fret,iso_efret = rigid_fret(geom)
	
	#Calculate k2, FRET for hemi-cylinder system
	cyl_efret,cyl_iso_fret,cyl_std,cyl_iso_std = cyl_fret(geom)

	#Calculate efret for base hemi-cylinder system
	base_efrets,base_std = base_fret(geom)

	#Save values
	print("Rigid fret:",rig_fret)
	print("iso fret:",iso_efret)
	new_name = "%s%s.dat" % (args.c3s[:-4],args.c5s[:-4])
	np.savetxt(new_name,np.array([[rig_fret,iso_efret,cyl_efret,cyl_iso_fret,base_efrets]]))

def load_dna(fn):
	"""Load file topology.

	Parameters:
		fn: *str*
			Path to DNA pdb made by 3DNA.
	Returns:
		dna_xyz: *np.array*
			Output of md.load_pdb()
		dna_top *md.Topology*
			dna.top.
	"""
	dna = md.load_pdb(fn)
	return dna.xyz[0], dna.top

def load_seqs(c3s,c5s):
	"""Load strand sequence. Each strand sequence file contains one integer,
	which corresponds to the C3 or C5 location read 5' --> 3' in the
	corresponding DNA strand.

	Parameters:
		c3s: *str*
			Path to cy3* sequence file.
		c5s: *str*
			Path to cy5 sequence file.
	Returns:
		cy3_seq: *int*
			Integer index of Cy3.
		cy5_seq: *int*
			Integer index of Cy5.
	"""
	cy3_seq = np.loadtxt(c3s, dtype=int)
	cy5_seq = np.loadtxt(c5s,dtype=int)
	cy5_seq -= 2
	return cy3_seq, cy5_seq

def calc_cy3_params(preb):
	"""Calculate minimal description of dye orientation based on current file.
	
	Parameters:
		preb: *str*
			Path to prebuilt atomistic DNA file..
	Returns:
		link_len: *float*
			Length from COM of phosphates to COM of post-linker dye
	"""
	dna = md.load_pdb(preb)
	xyz = dna.xyz[0]
	top = dna.top
	
	P1 = np.array([11.03,-19.944,-13.977])
	P2 = np.array([6.583,-15.62,-16.446])
	cent_c = np.array([4.147,-22.933,-16.243])

	cent_p = np.average((P1,P2),axis=0)
	link_len = np.linalg.norm(cent_c - cent_p)

	link_len /= 10
	#print("link_len:",link_len)
	return link_len

def calc_geom(dna_xyz,dna_top,cy3_seq,cy5_seq,link_len):
	"""Calculate Rigid fret if FRET molecule points directly out from DNA core.
	
	Parameters:
		dna_xyz: *np.array*
			Output of md.load_pdb()
		dna_top *md.Topology*
			dna.top.
		cy3_seq: *int*
			Integer index of Cy3, from 0.
		cy5_seq: *int*
			Integer index of Cy5, from 0.
		link_len: *float*
			Length from COM of phosphates to COM of post-linker dye
	Returns:
	geom *dict*
	Dictionary with entries:
		cy3_vect: *np.array*
			Instantaneous cy3 vector
		cy3_com: *np.array*
			Instantaneous cy3 COM
		cy5_vect: *np.array*
			Instantaneous cy5 vector
		cy5_com: *np.array*
			Instantaneous Cy5 COM
		cy3_p_com: *np.array*
			COM of phosphates for Cy3.
		cy3_outward_vec: *np.array*
			Vector pointing outward from p_com for cy3.
		cy5_p_com: *np.array*
			COM of phosphates for Cy5.
		cy5_outward_vec: *np.array*
			Vector pointing outward from p_com for cy5.
	"""
	geom = dict()
	#Determine phosphates of Cy3, Cy5
	xyz = dna_xyz

	chains = [ch for ch in dna_top.chains]
	
	cy3_ch = chains[0]
	cy5_ch = chains[1]

	#Get phosphates at beginning and end of bp
	all_residues = [r for r in cy3_ch.residues]
	prev_res = all_residues[cy3_seq-1]
	cy3_res = all_residues[cy3_seq]
	phosphates = []
	#Save xyz value
	for at in prev_res.atoms:
		if at.name=="P":
			phosphates.append(xyz[at.index])
	for at in cy3_res.atoms:
		if at.name=="P":
			phosphates.append(xyz[at.index])
	
	#Treat vector from phosphate center to base COM as "inward" vector
	base_ats = [at for at in cy3_res.atoms if "'" not in at.name]
	base_inds = np.array([at.index for at in base_ats],dtype=int)
	base_com = np.average(dna_xyz[base_inds],axis=0)
	#print("Base com:",base_com)
	p0_pos = phosphates[0]
	p1_pos = phosphates[1]
	p_com = np.average((p0_pos,p1_pos),axis=0)
	geom['cy3_p_dist'] = np.linalg.norm(p0_pos - p1_pos)
	geom['cy3_p_com'] = np.copy(p_com)
	cy3_vect = p1_pos - p0_pos
	cy3_vect /= np.linalg.norm(cy3_vect)
	
	#Calculate cy3 COM
	#outward_vec = (p_com - base_com) / np.linalg.norm(p_com - base_com)
	#Assume outward vec is perpendicular to z axis
	outward_vec = np.cross(p1_pos-p0_pos,np.array([0,0,1]))
	outward_vec /= np.linalg.norm(outward_vec)
	cy3_com = link_len * outward_vec + p_com
	geom['cy3_outward_vec'] = np.copy(outward_vec)	

	#Reperform above for cy5	
	all_residues = [r for r in cy5_ch.residues]
	prev_res = all_residues[cy5_seq-1]
	cy5_res = all_residues[cy5_seq]
	phosphates = []
	#Save xyz value
	for at in prev_res.atoms:
		if at.name=="P":
			phosphates.append(xyz[at.index])
	for at in cy5_res.atoms:
		if at.name=="P":
			phosphates.append(xyz[at.index])
	
	#Treat vector from phosphate center to base COM as "inward" vector
	base_ats = [at for at in cy5_res.atoms if "'" not in at.name]
	base_inds = np.array([at.index for at in base_ats],dtype=int)
	base_com = np.average(dna_xyz[base_inds],axis=0)
	#print("Base com:",base_com)
	p0_pos = phosphates[0]
	p1_pos = phosphates[1]
	geom['cy5_p_dist'] = np.linalg.norm(p0_pos - p1_pos)
	p_com = np.average((p0_pos,p1_pos),axis=0)
	cy5_vect = p1_pos - p0_pos
	cy5_vect /= np.linalg.norm(cy5_vect)
	
	#Calculate cy3 COM
	#outward_vec = (p_com - base_com) / np.linalg.norm(p_com - base_com)
	outward_vec = np.cross(p1_pos-p0_pos,np.array([0,0,1]))
	outward_vec /= np.linalg.norm(outward_vec)
	cy5_com = link_len * outward_vec + p_com
	
	geom['cy3_vect'] = np.copy(cy3_vect)
	geom['cy3_com'] = np.copy(cy3_com)
	geom['cy5_vect'] = np.copy(cy5_vect)
	geom['cy5_com'] = np.copy(cy5_com)
	geom['cy5_p_com'] = np.copy(p_com)
	geom['cy5_outward_vec'] =np.copy(outward_vec)
	geom['link_len'] = link_len
	d = np.linalg.norm(geom['cy5_p_com'] - geom['cy3_p_com'])
	print("geom dist:",d)
	with open("geom_distance",'a') as f:
		f.write("%s\n" % np.round(d,4))
	return geom
	

def rigid_fret(geom):
	"""Calculate Rigid fret if FRET molecule points directly out from DNA core.
	
	Parameters:
		geom: *dict*
			Output from calc_geom.
	Returns:
		rigid_k2: *float*
			Instantaneous k2 for single position.
		rigid_fret: *float*
			Instantaneous FRET for single position.
		iso_fret: *float*
			Instantaneous FRET assuming isotropic motion.
	"""
	

	R = (geom['cy5_com'] - geom['cy3_com'])
	#R = geom['cy5_p_com'] - geom['cy3_p_com']
	r = R/np.linalg.norm(R)
	#Apply K2 formula
	#cy3 is donor
	cos_t = np.dot(geom['cy5_vect'],geom['cy3_vect'])
	cos_d = np.dot(r,geom['cy3_vect'])
	cos_a = np.dot(r,geom['cy5_vect'])
	k2 = (cos_t - 3*cos_d*cos_a)**2
	#r0 = 52.68
	r0 = geom['r0']

	efret = 1/(1 + (np.linalg.norm(R)/(r0*k2/(2/3)))**6)
	print("Distance:",np.round(np.linalg.norm(R),2))
	with open("probe_distance",'a') as f:
		f.write("%s\n" % np.round(np.linalg.norm(R),4))
	iso_efret = 1/(1+(np.linalg.norm(R)/r0)**6)
	
	return k2,efret,iso_efret

def cyl_fret(geom,n=100):
	"""Calculate fret if FRET molecule lies on the outside of the hemi-cylinder.
	Integrate numerically via 100-point gaussian quadrature.
	
	Parameters:
		geom: *dict*
			Output from calc_geom.
		n: *int*
			Order of quadrature. Default = 100.
	Returns:
		cyl_fret: *float*
			Instantaneous FRET for single position.
		iso_cyl_fret: *float*
			Instantaneous FRET assuming isotropic motion.
		cyl_std: *float*
			stdev of cyl fret.
		iso_cyl_std: *float*
			stdev of cyl isofret.
	"""
	upward_cy3 = np.cross(geom['cy3_outward_vec'],geom['cy3_vect'])
	upward_cy5 = np.cross(geom['cy5_outward_vec'],geom['cy5_vect'])
	upward_cy3 /= np.linalg.norm(upward_cy3)
	upward_cy5 /= np.linalg.norm(upward_cy5)
	points,weights = np.polynomial.legendre.leggauss(n)
	thetas = (points + 1)/2*np.pi
	
	cy3_pos = []
	for i in range(n):
		cy3_pos.append(geom['cy3_p_com'] + (geom['link_len']*np.cos(thetas[i])*upward_cy3 + 
		geom['link_len']*np.sin(thetas[i])*geom['cy3_outward_vec']))

	cy5_pos = []
	for i in range(n):
		cy5_pos.append(geom['cy5_p_com'] + (geom['link_len']*np.cos(thetas[i])*upward_cy5 + 
		geom['link_len']*np.sin(thetas[i])*geom['cy5_outward_vec']))
	
	#Calculate rigid fret:
	Rs = np.array([cy3_pos[i] - cy5_pos[i] for i in range(n)])
	#Rs = np.array([geom['cy5_p_com'] - geom['cy3_p_com']]*n)
	rs = [R/np.linalg.norm(R) for R in Rs]
	#r0 = 52.68
	r0 = geom['r0']
	cos_t = np.dot(geom['cy5_vect'],geom['cy3_vect'])
	cos_ds = [np.dot(r,geom['cy3_vect']) for r in rs]
	cos_as = [np.dot(r,geom['cy5_vect']) for r in rs]
	k2s = np.array([(cos_t - 3*cos_ds[i]*cos_as[i])**2 for i in range(n)])
	
	efrets = np.array([1/(1 + (np.linalg.norm(Rs[i])/(r0*k2s[i]/(2/3)))**6) for i in range(n)])
	iso_efrets = np.array([1/(1 + (np.linalg.norm(Rs[i])/r0)**6) for i in range(n)])
	avg_efrets = np.average(efrets,weights=weights)
	avg_iso_efrets = np.average(iso_efrets,weights=weights)
	#Calculate weighted stdevs manually
	std_efrets = np.sqrt(np.average((efrets-avg_efrets)**2,weights=weights))
	std_iso_efrets = np.sqrt(np.average((iso_efrets-avg_iso_efrets)**2,weights=weights))
	
	return avg_efrets,avg_iso_efrets,std_efrets,std_iso_efrets

def base_fret(geom,n=50000):
	"""Calculate fret if FRET molecule lies on the bases of the hemi-cylinder.
	Integrate numerically via monte carlo method.
	
	Parameters:
		geom: *dict*
			Output from calc_geom.
		n: *int*
			Order of quadrature. Default = 100.
	Returns:
		base_fret: *float*
			Instantaneous FRET for single position.
		base_std: *float*
			stdev of cyl fret.
	"""
	upward_cy3 = np.cross(geom['cy3_outward_vec'],geom['cy3_vect'])
	upward_cy5 = np.cross(geom['cy5_outward_vec'],geom['cy5_vect'])
	upward_cy3 /= np.linalg.norm(upward_cy3)
	upward_cy5 /= np.linalg.norm(upward_cy5)

	#Create random points with equal weight
	points = np.random.rand(n,8)
	#Columns: theta1, theta2, r1, r2.then cy5: theta1, theta2, r1, r2.
	points[:,:2] *= np.pi
	points[:,4:6] *= np.pi
	points[:,2:4] *= geom['link_len']
	points[:,6:] *= geom['link_len']
	

	cy3_pos = []
	cy3_vect = []
	for i in range(n):
		p1 = geom['cy3_p_com'] + (points[i,2]*np.cos(points[i,0])*upward_cy3+
			points[i,2]*np.sin(points[i,0])*geom['cy3_outward_vec'])
		p2 = geom['cy3_p_com'] + (points[i,3]*np.cos(points[i,1])*upward_cy3+
			points[i,3]*np.sin(points[i,1])*geom['cy3_outward_vec'])
		cy3_pos.append((p1 + p2)/2)
		v = geom['cy3_vect']*geom['cy3_p_dist'] + (p2-p1)
		cy3_vect.append(v / np.linalg.norm(v))

	cy5_pos = []
	cy5_vect = []
	for i in range(n):
		p1 = geom['cy5_p_com'] + (points[i,2]*np.cos(points[i,0])*upward_cy3+
			points[i,2]*np.sin(points[i,0])*geom['cy5_outward_vec'])
		p2 = geom['cy5_p_com'] + (points[i,3+4]*np.cos(points[i,1+4])*upward_cy3+
			points[i,3+4]*np.sin(points[i,1+4])*geom['cy5_outward_vec'])
		cy5_pos.append((p1 + p2)/2)
		v = geom['cy5_vect']*geom['cy5_p_dist'] + (p2-p1)
		cy5_vect.append(v / np.linalg.norm(v))
	
	#Calculate rigid fret:
	Rs = np.array([cy3_pos[i] - cy5_pos[i] for i in range(n)])
	#Rs = np.array([geom['cy5_p_com'] - geom['cy3_p_com']]*n)
	rs = [R/np.linalg.norm(R) for R in Rs]
	#r0 = 52.68
	r0= geom['r0']
	cos_ts = [np.dot(cy5_vect[i],cy3_vect[i]) for i in range(n)]
	cos_ds = [np.dot(rs[i],cy3_vect[i]) for i in range(n)]
	cos_as = [np.dot(rs[i],cy5_vect[i]) for i in range(n)]
	k2s = np.array([(cos_ts[i] - 3*cos_ds[i]*cos_as[i])**2 for i in range(n)])
	
	efrets = np.array([1/(1 + (np.linalg.norm(Rs[i])/(r0*k2s[i]/(2/3)))**6) for i in range(n)])
	avg_efrets = np.average(efrets)
	#Calculate weighted stdevs manually
	std_efrets = np.std(efrets)
	return avg_efrets,std_efrets


def load_dna_xyz():
	"""Perform all operations in box 3, returning only necessary data.

	Returns:
			fvu0: *np.array*
			fvu1: *np.array*
			dna_ref_xyz: *np.array*
			nuc_ref_xyz: *np.array*
			dna_type_arr: *np.array*
			nuc_type_arr: *np.array*
			all_types
			type_inds
			
	"""

	#############BOX 3
	####Loading Reference DNA

	dna_path="./3bp_dna.pdb"
	dna_traj=md.load_pdb(dna_path)
	dna_traj=dna_traj.center_coordinates()
	dna_ref_xyz = dna_traj.xyz[0]
	dna_types = dna_traj.top.to_dataframe()[0]["name"]

	#print("Dna_ref_xyz: ",dna_ref_xyz)
	#print("Dna_ref_xyz.shape: ",dna_ref_xyz.shape)

	###Loading Reference Nucleosome
	nuc_path="./nuc.pdb"
	nuc_traj=md.load_pdb(nuc_path)
	nuc_traj=nuc_traj.center_coordinates()
	nuc_ref_xyz = nuc_traj.xyz[0]
	nuc_types = nuc_traj.top.to_dataframe()[0]["name"]

	#Create a sorted list of all types
	all_types = set(dna_types).union(set(nuc_types))
	all_types = list(all_types)
	all_types.sort()
	type_inds = np.arange(len(all_types),dtype=int)

	#Create a type dict
	type_dict = dict((t,ind) for t, ind in zip(all_types,type_inds))

	#Create reference type lists
	nuc_type_arr = np.array([type_dict[t] for t in nuc_types],dtype = int)
	dna_type_arr = np.array([type_dict[t] for t in dna_types],dtype=int)

	fvu0 = Quaternion(matrix=np.eye(3))
	fvu1 = Quaternion(matrix=np.eye(3))

	#Rotate the nuc_ref_xyz to match the 1CPN standard
	#f_quat = tu2rotquat(np.pi,np.array([0,0,1]))
	corr_quat = tu2rotquat(-np.pi/2,np.array([0,1,0]))
	
	#q = quat_multiply(corr_quat,f_quat)#f_quat,corr_quat)
	#q = f_quat
	second_quat = tu2rotquat(-np.pi/2,np.array([1,0,0]))
	#q = quat_multiply(q,second_quat)
	q = quat_multiply(second_quat,corr_quat)
	#q = quat_multiply(corr_quat,second_quat)
	f_quat = tu2rotquat(np.pi,np.array([1,0,0]))
	q = quat_multiply(f_quat,q)
	#q = corr_quat
	print("q:",q)
	nuc_ref_xyz = np.apply_along_axis(lambda xyz: quat_vec_rot(xyz,q),1,nuc_ref_xyz)	

	#corr_quat = Quaternion(corr_quat)

	#nuc_ref_xyz = np.apply_along_axis(corr_quat.rotate, 1, nuc_ref_xyz)
	print("Corrected 1kx5 rotation.")
	return fvu0, fvu1, dna_ref_xyz, nuc_ref_xyz, dna_type_arr, nuc_type_arr, all_types, type_inds

	#################


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-f','--c3s', type=str, default=None,
		help='Cy3-bound strand file', required = True)
	parser.add_argument('-s','--c5s', type=str, default=None,
		help='Cy5-bound strand file', required = True)
	parser.add_argument('-b','--base', type=str, default=None,
		help='Curved DNA base pdb', required = True)
	parser.add_argument('-p','--preb', type=str, default=None,
		help='Prebuilt Cy3 structure pdb', required = True)

	args = parser.parse_args()
	
	main()
