# MD-active-subspace

Jupyter notebooks to replicate the results of 

*Global ranking of the sensitivity of interaction potential contributions within classical molecular dynamics force fields*, (submitted), 2023.

**Abstract**
Uncertainty quantification (UQ) is rapidly becoming a sine qua non for all forms of computational science out of which actionable outcomes are anticipated. While UQ is routinely performed for many kinds of large scale engineering applications within industry, for weather and climate forecasting, alongside socio-political and other human affairs, much of the microscopic world of atoms and molecules has seemingly remained immune to these developments. However, since fundamental problems of reproducibility and reliability are also implicated, it is essential that practitioners of these long-established methods pay attention to the issues concerned. In the present paper, we undertake a full scale, global UQ study of all of the epistemic and aleatoric contributions to key quantities of interest that are calculated in typical classical molecular dynamics simulations with physicochemical specificity including material properties and binding free energy predictions in drug discovery and personalised medicine. Using scalable UQ methods based on active subspaces that invoke machine learning and Gaussian processes, we globally rank the sensitivity of the calculated quantities. The analyses reveal that the uncertainty is dominated by only a small number of the many hundreds of interaction potential parameters within the force fields employed. This ranking is of immediate scientific interest as it highlights what forms of interaction control the uncertainty of the predictions and enable systematic improvements to be made in future optimisations of such parameters.

## Jupyter notebooks

* `epoxy/epoxy.ipynb`: notebook to replicate the results of the epoxy-resin application.
* `esmacs/esmacs.ipynb`: notebook to replicate the results of the ESMACS binding free energy application.
* `ties/ties.ipynb`: notebook to replicate the results of the TIES relative binding free energy application.

## Data

The MD training data can be found in the three application directories, stored in CSV format.

## Funding

The authors acknowledge funding support from (i) the UK EPSRC for the UK High-End Computing Consortium (EP/R029598/1), the Software Environment for Actionable \& VVUQ-evaluated Exascale Applications (SEAVEA) grant (EP/W007762/1), the UK Consortium on Mesoscale Engineering Sciences (UKCOMES grant no. EP/L00030X/1), and the Computational Biomedicine at the Exascale (CompBioMedX) grant (EP/X019276/1); (ii) the UK MRC Medical Bioinformatics project (grant no. MR/L016311/1); (iii) the European Commission for EU H2020 CompBioMed2 Centre of Excellence (grant no. 823712) and EU H2020 EXDCI-2 project (grant no. 800957). We made use of SuperMUC-NG at Leibniz Supercomputing Centre under project COVID-19-SNG1, and the ARCHER2 UK National Supercomputing Service under project **XXX**.
