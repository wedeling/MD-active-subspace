# MD-active-subspace

Jupyter notebooks to replicate the results of:

Wouter Edeling, Maxime Vassaux, Yiming Yang, Shunzhou Wan, Peter Coveney, *Global ranking of the sensitivity of interaction potential contributions within classical molecular dynamics force fields*, (submitted), 2023.

**Abstract**
Uncertainty quantification (UQ) is rapidly becoming a sine qua non for all forms of computational science out of which actionable outcomes are anticipated. While UQ is routinely performed for various large-scale meteorological, climate and engineering applications, such as for weather and climate forecasting, much of the microscopic world of atoms and molecules has seemingly remained immune to these developments. However, due to the fundamental problems of reproducibility and reliability, it is essential that practitioners pay attention to the issues concerned. Here a UQ study is undertaken of classical molecular dynamics with a particular focus on uncertainties in the high-dimensional force-field parameters, which affect key quantities of interest, including material properties and binding free energy predictions in drug discovery and personalized medicine. Using scalable UQ methods based on active subspaces that invoke machine learning and Gaussian processes, the sensitivity of the input parameters is ranked. The analyses reveal that the prediction uncertainty is dominated by a small number of the hundreds of interaction potential parameters within the force fields employed. This ranking is of immediate scientific interest as it highlights what forms of interaction control the uncertainty of the predictions, and enables systematic improvements to be made in future optimizations of such parameters.

## Jupyter notebooks

* `epoxy/epoxy.ipynb`: notebook to replicate the DAS results of the epoxy-resin application.
* `epoxy/epoxy_gp.ipynb`: notebook to replicate the KAS-GP results of the epoxy-resin application.
* `esmacs/esmacs.ipynb`: notebook to replicate the DAS results of the ESMACS binding free energy application.
* `esmacs/esmacs_gp.ipynb`: notebook to replicate the KAS-GP results of the ESMACS binding free energy application.
* `ties/ties.ipynb`: notebook to replicate the DAS results of the TIES relative binding free energy application.
* `ties/ties_gp.ipynb`: notebook to replicate the KAS-GP results of the TIES relative binding free energy application.

## Data

The MD training data can be found in the three application directories, stored in CSV format.

## Funding

The authors acknowledge funding support from (i) the UK EPSRC for the UK High-End Computing Consortium (EP/R029598/1), the Software Environment for Actionable \& VVUQ-evaluated Exascale Applications (SEAVEA) grant (EP/W007762/1), the UK Consortium on Mesoscale Engineering Sciences (UKCOMES grant no. EP/L00030X/1), and the Computational Biomedicine at the Exascale (CompBioMedX) grant (EP/X019276/1); (ii) the UK MRC Medical Bioinformatics project (grant no. MR/L016311/1); (iii) the European Commission for EU H2020 CompBioMed2 Centre of Excellence (grant no. 823712) and EU H2020 EXDCI-2 project (grant no. 800957). We made use of SuperMUC-NG at Leibniz Supercomputing Centre under project COVID-19-SNG1, and the ARCHER2 UK National Supercomputing Service under the SEAVEA grant (EP/W007762/1).
