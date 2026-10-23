# SOPA 2027 — SHAllow RADar Open Processing Attempt (modernized)

SOPA is a suite of open source algorithms for SHARAD (SHAllow RADar) low-level
data processing. This tree, **SOPA_2027**, is the modernized version of the
suite: same algorithms, same scientific outputs, brought to current coding
standards step by step.

The non-negotiable constraint of the whole modernization: **every ENVI product
SOPA_2027 writes must be byte-identical to what SOPA_2026 writes** — verified
mechanically by the byte-fidelity gate, never by eye. No change ships without
a passing gate.

- `SOPA_2026/` (sibling directory) is the frozen reference. It is never
  modified; its integrity is enforced by a checksum manifest.
- `AGENTS.md` (workspace root) and the `wiki/` describe the working model,
  the gate, and the current phase — read them before changing anything here.

## Status

Phase 0 (2026-10-23): SOPA_2027 is a verbatim clone of SOPA_2026 — no source
changes yet. Fidelity gate: **PASS** (16/16 files byte-identical across the
4-configuration matrix). The phase plan lives in
`wiki/state/project-status.md`.

## Byte-fidelity gate

From the workspace root (`/mnt/SOPA`):

```bash
python3 tools/fidelity/fidelity_gate.py [--reuse-reference]
```

The gate runs both codebases over a 4-configuration matrix (standard, wider
aperture, squint +1°, squint −1°) on real SHARAD data (PDS EDR orbit 1260201 +
MOLA128), then requires sha256-identical ENVI products. Details, matrix, and
failure policy: `wiki/architecture/fidelity-gate.md`.

## Requirements

Identical to SOPA_2026 — pinned versions are part of the fidelity contract
(numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2, Python 3.14.4; see
`requirements.txt`). Dependency upgrades are separate, gate-proven tasks.

## Virtual environment

`/home/ferroa/venvs/sopa_2027` (outside `/mnt/SOPA`: VirtualBox shared folders
do not support symlinks). Recreating it (`python3.14-venv` is not installed on
this machine):

```bash
python3 -m venv --without-pip /home/ferroa/venvs/sopa_2027
/home/ferroa/venvs/sopa_2026/bin/python -m pip --python /home/ferroa/venvs/sopa_2027/bin/python install -r requirements.txt
```

## Running

```bash
source /home/ferroa/venvs/sopa_2027/bin/activate
cd /mnt/SOPA/SOPA_2027
python run_sopa_processing_2610.py    # full example (hours, single core)
python tests/real_data_test.py        # real data, reduced range (~2 min)
python tests/smoke_test.py            # fully synthetic (~1 min)
```

Plots are interactive (matplotlib Qt backend via PyQt6); headless runs select
the Agg backend automatically in the test scripts — set `MPLBACKEND=Agg`
yourself elsewhere.

## Provenance

SOPA was written by Adamo Ferro (2016–2019, `SOPA_OLD/`), ported to a current
stack in 2026 (`SOPA_2026/`), and is modernized as `SOPA_2027` under the
GPLv3 and the acknowledgment clause below.

---

# SOPA (original README, kept for reference)

SOPA is thought to be a suite of open source algorithms for SHAllow RADar (SHARAD) low-level data processing written in python v3.

At the time of writing the suite includes:
- data reading with common interface for:
  - PDS Italian EDR raw data (echoes and geometric data*);
  - PDS Italian RDR focused data (echoes and geometric data*)
  - CO-SHARPS raw data (echoes and full ancillary dataset);
  - PDS US RDR focused data (echoes and geometric data**);
- an "essential" version of SOFA (SHARAD Open Focusing Attempt, http://af-projects.it/sofa), a focuser that, besides standard nadir focusing, allows squinted synthetic aperture radar processing***;
- surface clutter simulators****:
  - dRSsim, a dummy radar sounder surface clutter simulator;
  - sfRSsim, an enhanced simulator using square facets that can use a cos^exp law to weight the effect of surfaces not facing the radar;
- plotting tools for quick visualization of focused radargrams, simulations, ground distance images, and first return maps superimposed to the DEM.

Unfortunately, almost no documentation is provided. An example program showing how to use the main functions can be found in the root of the project.

*Orbits extracted from PDS Italian data may be not sufficiently precise for data acquired before 2013.

**Orbits extracted from PDS US RDR data may be not sufficiently precise for supporting simulations and or focusing.

***By now, SOFA can handle only night time radargrams as no correction of ionosphere effects has been implemented yet.

****The simulation principles are similar to those described in http://dx.doi.org/10.1109/TGRS.2012.2219315.


Feel free to contact me for any help (contacts here: http://af-projects.it/contacts).

***NOTE:** This project has not been developed during my work at the University of Trento, nor it has been supported by the University of Trento in any way.*

### Requirements (original)
SOPA has been tested only in a Linux environment using python v3.5 and the following modules:
- sys
- math
- os
- glob (needed only for parsing CO-SHARPS data)
- datetime
- gzip (needed only if txt files related to CO-SHARPS ancillary data have been downloaded gzipped)
- numpy 1.17.4
- scipy 1.3.2
- bitstring (needed only for PDS Italian EDR data)
- ray 0.7.3 (needed only for parallel computation in surface clutter simulators and SOFA, not mandatory)
- matplotlib 3.0.3 (needed only for plotting classes)

### Current version
0.02

### Disclaimer

The author does not guarantee that this software will always provide correct results nor that it will not crash your hardware. In any case, any use of SOPA is ONLY user responsibility. The use of SOPA or part of it for the creation of any sub-product (e.g., scientific papers, posters, images, other softwares) must be acknowledged.
As you may imagine, the code is not optimized for fast running.
