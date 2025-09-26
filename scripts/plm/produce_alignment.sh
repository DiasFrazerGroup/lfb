#!/bin/bash
set -eou pipefail

# Path to mmseqs database to search from
DB_PATH=$1
# Path to input sequence, in fasta format
INPUT_FASTA_FILE=$2
# Output file path
OUTPUT_TSV_FILE=$3
# Number of search iterations to run
NUM_ITERATIONS=5
# Number of threads
THREADS=10
# Memory limit for mmseqs search
SPLIT_MEMORY_LIMIT=30
# Temporary directory
TEMP_DIR="./tmp_lfb_alignment/"

# Create temporary directory
mkdir -p $TEMP_DIR

# Create query database
mmseqs createdb "$INPUT_FASTA_FILE" "$TEMP_DIR/query_db"

# Regular search
mmseqs search "$TEMP_DIR/query_db" "$DB_PATH" "$TEMP_DIR/result_db" "$TEMP_DIR" \
    -s 7.5 \
    --threads $THREADS \
    --split-memory-limit ${SPLIT_MEMORY_LIMIT}G \
    --num-iterations $NUM_ITERATIONS \
    -a

# Get alignment tsv file
mmseqs convertalis "$TEMP_DIR/query_db" "$DB_PATH" "$TEMP_DIR/result_db" \
    "$OUTPUT_TSV_FILE" \
    --format-output "query,target,qstart,qend,tstart,tend,cigar,qseq,tseq"

# Clean up temporary files
rm -rf $TEMP_DIR/*

echo "> Completed processing $INPUT_FASTA_FILE"
