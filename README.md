# QG

The base QG model on arbitrary geometries with homogeneous boundary conditions implemented in Pytorch is presented in [https://doi.org/10.5194/gmd-17-1749-2024](https://doi.org/10.5194/gmd-17-1749-2024) (accepted at Geoscientific Model Development). The original code for the homogeneous model is available at https://github.com/louity/MQGeometry.

## Additions

### Inhomogeneous boundaries

The code has been modified to deal with inhomogeneous boundary conditions, on regular geometries only at the moment.

### Modified assimilation models

The RGSI model has been implemented, along with a "Forced" model.