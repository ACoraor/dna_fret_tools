#!/bin/bash
rm probe_distance
rm geom_distance
rm all_data.dat; for n in {6..20}; do python dna_fret.py -f A.seq -s B${n}.seq -b 2dye_construct.pdb -p dna_20bp_with_cy3_only.pdb; cat AB${n}.dat >> all_data.dat; done
