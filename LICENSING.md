# Licensing, attribution and citation

PRT-DeepONet Studio, version 1.0. Released 9 August 2026.

This file explains who wrote what, under which terms each part may be used,
and how to cite the work. The full legal text of the GNU General Public
License, version 3, is in `LICENSE`.

---

## Who wrote what

### The three-dimensional work, the simulation generators and this application

| | | |
|---|---|---|
| Shahram Asgari | author | shahram.asgari@uga.edu |
| Christof Meile | principal investigator | cmeile@uga.edu |

Meile Lab, Department of Marine Sciences, University of Georgia, Athens,
Georgia 30602, USA

Covering the three-dimensional formulation, the geometry generator, the D2Q9
and D3Q19 lattice Boltzmann flow solvers, the shared advection, diffusion and
reaction solver, the CompLaB3D campaign and collection tools, the training,
evaluation and prediction pipeline, the three switches, and the desktop
application.

Copyright (C) 2026 Shahram Asgari and Christof Meile, Meile Lab, University of
Georgia.

### The two-dimensional work

PRT-DeepONet, on which the two-dimensional side of this project builds:

| | |
|---|---|
| Yehoon Kim | wnsla7323@naver.com |
| Heewon Jung | hjung@cnu.ac.kr |

Jung Lab, Chungnam National University, Republic of Korea.

Copyright (C) 2025 Jung Lab. Licensed under the GNU General Public License,
version 3 or later.

Their release is included in `2D/` **unmodified** and is treated as read-only.
It supplies 3000 pore domains, trained weights for three reaction types
(irreversible sorption, reversible sorption and Monod kinetics), and the
notebooks that run them. No file in `2D/` is written to by any part of this
project: every reference to it in our code is a read, a file-open dialog, or a
menu item that opens the folder.

Their architecture, a convolutional branch for the geometry, a fully connected
branch for the dimensionless numbers, and a trunk taking position, time and
the geodesic distance, is the starting point for the three-dimensional
extension. Where this project reuses their trained weights it does so by
reading the published parameter files. It does not copy or modify their
source.

Their licence text is also distributed with their code, at `2D/LICENSE`.

---

## Licence

### The two-dimensional release

Everything in `2D/` keeps its own licence. GPL v3 applies to it unchanged, and
is distributed with it.

### The three-dimensional code and this application

Offered under a **dual licence**.

**Academic, research and teaching use.** GNU General Public License, version 3
or later. Free to use, study, modify and redistribute, provided that
derivative work carries the same licence and stays open. The full text is in
`LICENSE`, and a copy with no header is in `COPYING`.

**Commercial and industrial use.** Not covered by the above. A separate
licence is required. Write to the addresses at the top of this file.

### Two open questions

Both of these are worth knowing now rather than discovering later, and neither
has yet been put to anyone qualified to answer it. Both should be, before this
is distributed outside the group.

Dual licensing requires the agreement of every copyright holder, and work
produced at a university is usually owned in part by the institution. A
commercial licence would therefore go through the University of Georgia's
research and innovation office rather than being granted informally.

Separately, whether a model warm-started from GPL-licensed trained weights is
itself a derivative work is an unsettled question in law.

---

## How to cite

**This project:**

> Asgari, S., Meile, C., 2026. PRT-DeepONet Studio, version 1.0. Meile Lab,
> Department of Marine Sciences, University of Georgia, Athens, Georgia, USA.
> Contact: shahram.asgari@uga.edu (author), cmeile@uga.edu (principal
> investigator).

**The two-dimensional work it builds on:**

> Kim, Y., Jung, H., 2025. PRT-DeepONet. Jung Lab, Chungnam National
> University, Republic of Korea. Contact: wnsla7323@naver.com,
> hjung@cnu.ac.kr.

If you use both, cite both. The geometry-aware three-dimensional method
extends their architecture; it does not replace it.

A machine-readable version of the first entry is in `CITATION.cff`, which is
what puts the "Cite this repository" button on the GitHub page.

---

## Verifying an installation

**Set up, then Check everything at once** runs ten independent groups of
checks, including both simulators against analytic answers: plug flow must
leave exp(-Da) at the outlet, and a front must spread as wide as the exact
complementary error function solution says it should.

**Tools, then Where is everything?** lists every path the program uses.
