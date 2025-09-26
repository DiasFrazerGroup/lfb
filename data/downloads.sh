#!/bin/bash

WD=$(pwd)

# -------------------------------------------------------------------------------------------------
# Download the ProGen2 models

for model in progen2-large progen2-xlarge progen2-medium progen2-small; do
    wget -P ./data/progen2/${model} https://storage.googleapis.com/sfr-progen-research/checkpoints/${model}.tar.gz
    tar -xvf ./data/progen2/${model}/${model}.tar.gz -C ./data/progen2/${model}/
done

# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
# Clone ProteinGym
cd git 
git clone git@github.com:OATML-Markslab/ProteinGym.git
# Commit hash: e1d603d28ed8c6a27959c993de30312a83203a16
cd - 
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
# Download the ProteinGym data - v1.1
cd data
wget https://zenodo.org/api/records/13936340/files-archive
unzip files-archive && rm files-archive
unzip ProteinGym_v1.1.zip
cd -

# Unpack the ProteinGym data
cd data/ProteinGym_v1.1

# Get clinical data
mkdir clinical_ProteinGym_substitutions
unzip clinical_ProteinGym_substitutions.zip -d clinical_ProteinGym_substitutions && rm clinical_ProteinGym_substitutions.zip

mkdir clinical_msa_files
unzip clinical_msa_files.zip -d clinical_msa_files && rm clinical_msa_files.zip

# Get DMS data
unzip DMS_ProteinGym_substitutions.zip && rm DMS_ProteinGym_substitutions.zip
unzip DMS_msa_files.zip && rm DMS_msa_files.zip

# # Get model scores
unzip zero_shot_substitutions_scores.zip && rm zero_shot_substitutions_scores.zip
unzip zero_shot_clinical_substitution_scores.zip && rm zero_shot_clinical_substitution_scores.zip

cd $WD
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
# Download Zoonomia whole genome alignments

# See: https://cglgenomics.ucsc.edu/november-2023-nature-zoonomia-with-expanded-primates-alignment/

cd data
wget https://cgl.gi.ucsc.edu/data/cactus/447-mammalian-2022v1.hal
cd -

# -------------------------------------------------------------------------------------------------