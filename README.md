# RivMAP

RivMAP (**River Morphodynamics from Analysis of Planforms**) is a MATLAB toolbox for analyzing river planforms from binary channel masks.

This repository packages the original RivMAP MATLAB code for public visibility and reuse. It is being preserved as a first-class package, but it is **not actively maintained**.

## What RivMAP does

Given one or more binary channel masks, RivMAP provides tools to compute:

- channel centerlines
- banklines
- channel width (two methods)
- centerline direction and curvature
- centerline migration areas
- erosion and accretion areas
- cutoff detection and cutoff metrics
- channel-belt boundaries and centerlines
- spatial and temporal summaries of planform change

The package also includes a demo dataset and walkthrough based on annual channel masks from the Ucayali River in Peru.

## Repository layout

```text
RivMAP/
├── README.md
├── LICENSE
├── CITATION.cff
├── .gitignore
├── src/        # MATLAB functions
├── examples/   # demo script
├── data/       # small demo datasets
└── docs/       # walkthrough and notes
```

## Requirements

From the original MATLAB File Exchange release:

- MATLAB
- Image Processing Toolbox
- Mapping Toolbox for georeferenced image/vector stitching workflows

The File Exchange page lists the package as created with **R2015a** and compatible with any MATLAB release, but this repository does not currently provide a formal compatibility matrix.

## Quick start

1. Open MATLAB.
2. Change into the repository root.
3. Add the source directory to your path:
   ```matlab
   addpath(fullfile(pwd,'src'))
   ```
4. Run the demo:
   ```matlab
   run(fullfile('examples','DEMO.m'))
   ```

The demo script in this repository has been lightly updated to use paths relative to the repository root instead of a machine-specific local path.

## Included example data

This repository includes two demo data files:

- `data/riv.mat`
- `data/riv_georeffed.mat`

These are the same style of example assets used by the original File Exchange release and support the walkthrough in `examples/DEMO.m` and `docs/RivMAP Demo Walkthrough.docx`.

## Paper

If you use RivMAP in research, please cite the associated paper:

Schwenk, J., Khandelwal, A., Fratkin, M., Kumar, V., & Foufoula-Georgiou, E. (2017).  
**High spatiotemporal resolution of river planform dynamics from Landsat: The RivMAP toolbox and results from the Ucayali River.**  
*Earth and Space Science*, 4(2), 46–75.  
https://doi.org/10.1002/2016EA000196

## Provenance

RivMAP was originally released through MATLAB Central File Exchange as:

**RivMAP - River Morphodynamics from Analysis of Planforms**  
https://www.mathworks.com/matlabcentral/fileexchange/58264-rivmap-river-morphodynamics-from-analysis-of-planforms

This GitHub-ready version keeps the original MATLAB code largely intact while making a few packaging changes for repository use:

- MATLAB functions moved into `src/`
- demo script moved into `examples/`
- demo data moved into `data/`
- walkthrough document moved into `docs/`
- demo script updated to use relative paths

## Third-party components acknowledged in the original package

The original RivMAP package includes or references third-party MATLAB utilities, including:

- `interparc.m` (John D'Errico)
- `intersections.m` (Douglas M. Schwarz)
- `savfilt.m` (Vassili Pastushenko)
- `getkey` functionality used by `hand_clean.m` (Jos van der Geest)

Please retain the existing notices in `LICENSE` and the in-file headers if you redistribute modified versions.

## Status

This code is preserved here for visibility, reuse, and citation. It should be treated as an archived MATLAB toolbox rather than an actively developed software project.
