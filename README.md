# Codes, Data and Results for the Chromaticism paper



- As some data files are too large to push in Github, you can run the scripts to generate those data files. 

### Cleaned and preprocessed data:
- For the cleaned data (after preprocessing: removing un-annotated pieces, problematic parsing ... 
see the `Data/preprocess_logs/` under /Data/ for filtered pieces), they should be in the `Data/prep_data/` folder. 

Note: the file `processed_DLC_data.pickle` is too large to push in Github. 


### Data for analysis 
- The intermediate data for various analyses is under the folder `Data/prep_data/for_analysis/`.

  - The `chord_level_indices` contains both the dissonance and chromaticities for each chord in the DLC.
  - The file names of the other data files should be self-explanatory.


In general, in case any file is missing in the folder `Data/prep_data/for_analysis/` or 
the `processed_DLC_data.pickle` is missing, just run the script `Code/compute.py` to generate all the files.

### Results

Within the Results folder, you will find different sets of analyses in different folders.

The `correlation_analyses` and `GPR_analysis` also contains plotly figure (in html) that you can explore specific
datapoints. 

---
**Update May 9:**
3 sets of intermediate data and results are included in the repo based on the 3 versions of dissonance metric.

_Current decision_: we chose the version of normalized by chord size (figures and stats updated in manuscript).
See the following folder path for the data and results for this version:

- Data for analysis: `Data/prep_data/for_analysis/diss_version_NormByChordSize/`
- Results and figures: `Results/diss_version_NormByChordSize`