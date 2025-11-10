

# Overview

This repository contains the scripts used to analyze data and create figures for the manuscript "Non-canonical Genetic Markers Resolve the Pre-GOE Emergence of Aerobic Bacteria in Earth's History"


# Directory and Script Descriptions

If there are files missing, it might because it is too large. See [1007 bacterial proteomes](https://zenodo.org/records/17338761).

## python scripts

`load_data.py` and `general_func.py` are two python scripts that define some necessary functions and load in necessary data.

Note that path should be manually changed according to the repo path. 

## UsingDavinToEvaluate6Soft

Scripts for using five different predictors and GBDT40-LR

Timed Trees and Genome information excel retrieved from Davin 2025

`Final.combined.py` means how to collect all six predictor results and merged them into one

## Evaluate6Soft

Scripts for applying five different predictors on Madin dataset.

## trainging_sets

key tables of the Madin dataset. Also the phylophlan results.

## timing

RelTime setting files and the Input of the first step MCMCTree timing analysis.

## phyloSig

Scripts for running two analysis for identifying phyloSig of GBDT40.

## notebooks

IPython notebook used in this project.

## GTDB_results

Reltime Results and ACE probability, GBDT40_LR prediction results.

Some iTOL annotations files.


## Data

Source data for some figures.



# Notes
Within the scripts, if you find any import like `from bin.format_newick import renamed_tree`. Please referred to the other repo `https://github.com/444thLiao/evol_tk`.


Proteins used in Davin 2025 are deposited here. A [Zenodo Link](https://zenodo.org/records/17338761)

# Data availability


# Publication

Under review and submission.


# Contact Us
If you have any questions or suggestions on these scripts, you are welcome to contact us via email: l0404th@gmail.com.
If you have any questions on the paper and experimental parts, you are welcome to contact us via email: hluo2006@gmail.com.