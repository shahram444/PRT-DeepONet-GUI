# What the Python simulator can and cannot do

Generated from `settings_and_units.py --capabilities`. Regenerate it with:

    python 3D/tools/settings_and_units.py --capabilities > PYTHON_VS_COMPLAB.md

```

WHAT THE PYTHON SIMULATOR DOES, AND WHAT IT DOES NOT

These scripts are not a reimplementation of CompLaB and are not trying to be.
CompLaB is the reference: it is the one to publish a mechanism from. The
Python side exists so that a whole training dataset can be built on one
machine in an afternoon, which is what a neural operator needs and what a
cluster queue cannot give you. Knowing exactly where the two differ is the
difference between a fair comparison and a wasted month.

FLOW
  both          incompressible flow through the pore space, lattice Boltzmann,
                BGK collision, no-slip at the grain surfaces
  both          the no-slip wall sits halfway between the last fluid voxel and
                the first solid one, which is where Palabos puts it; this was
                checked against the analytic parabola and is in test_flow_solvers.py
  CompLaB only  MRT collision, multiple relaxation times
  CompLaB only  a flow field that is re-solved as the biofilm grows
  Python        D2Q9 in two dimensions, D3Q19 in three, single relaxation time,
                solved once per structure because the pore space is fixed

TRANSPORT
  both          advection, diffusion and reaction on the same grid as the flow
  both          Dirichlet feed at the inlet face, free outflow at the outlet,
                no flux at every grain surface
  both          a per-chemical diffusion coefficient
  CompLaB only  a separate diffusivity inside the biofilm (you can WRITE one
                here and it is read and reported, but the Python solver uses
                the pore value everywhere, because it carries no biofilm
                volume fraction to interpolate with)
  CompLaB only  Dirichlet conditions on the transverse faces
  Python        explicit time stepping with one combined stability limit; the
                run length comes from the field settling, not from a fixed
                step count

CHEMISTRY
  both          Monod kinetics, dual limitation on donor and acceptor
  both          biomass growth from the donor with a yield, and first-order
                decay
  both          an abiotic rate constant
  CompLaB only  aqueous speciation and chemical equilibrium, the R1-R73 tableau
  CompLaB only  precipitation, and pore voxels turning into solid
  CompLaB only  surface complexation, and mineral surface sites
  CompLaB only  many microbial pools, attached and planktonic, each with its
                own kinetics file
  Python        one reaction network: donor + acceptor -> product + biomass,
                with one attached biomass pool. Four chemicals. If your problem
                needs the fifth, it needs CompLaB.

COST, MEASURED ON ONE CORE
  Python        32^3 about 7 s per run, 48^3 about 58 s, 64^3 about 4 min,
                and the flow solve is once per structure
  CompLaB       hours per run, on a cluster, in a queue

WHICH TO USE
  Use the Python generator to build the training set, to sweep Peclet and
  Damkohler across decades, and to prove a pipeline end to end before spending
  cluster time. Use CompLaB for the reference runs the paper rests on, for any
  chemistry beyond one donor and one acceptor, and for anything where the pore
  space changes while the run proceeds.

  A network trained on Python data and then evaluated against CompLaB output
  is a fair test of the method. A network trained on Python data and reported
  as though it had seen CompLaB physics is not, and that is why every dataset
  file records which generator wrote it.

```

## Where your own numbers go

Write a settings file, edit it, check it, then hand it to either dataset builder:

```
python 3D/tools/settings_and_units.py --template work/settings.xml
python 3D/tools/settings_and_units.py --check    work/settings.xml
python 3D/tools/build_dataset_3d.py --settings work/settings.xml --out work/d.h5
```

In the window it is **Set up -> Simulation settings, in your own units**, and both dataset pages carry a *Settings file* box.
