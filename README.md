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

