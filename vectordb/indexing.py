import faiss
import numpy as np
import csv

from vectordb.config import SvlessVectorDBParams
from .centroids import CentroidMaster
import orjson
from lithops import Storage
from io import StringIO
import time
import pandas as pd
import multiprocessing
import logging

from collections import Counter
import statistics

def generate_index_blocks(id, obj, params, n_blocks, storage: Storage):
    """Generate an index from a CSV file"""
    faiss.omp_set_num_threads(6)
    ## Download Vectors
    start = time.time()
        
    csv_data = obj.data_stream.read().decode('utf-8')
    csv_buffer = StringIO(csv_data)
    csv_reader = csv.reader(csv_buffer)
    
    csv_reader = list(csv_reader)
    
    quotient, remainder = divmod(len(csv_reader), n_blocks)
    lower_elements = [quotient for i in range(n_blocks - remainder)]
    higher_elements = [quotient + 1 for j in range(remainder)]
    n_vecs_per_block = lower_elements + higher_elements
    
    i = 0
    key_id = id * n_blocks
    ids = []
    vectors = []

    for row in csv_reader:
        vector = row[1].split(" ")
        vector = [float(value) for value in vector if value != '']
        vectors.append(vector)
        ids.append(int(row[0]))

        if len(vectors) == n_vecs_per_block[i]:
            index = faiss.index_factory(params.features, f'IVF{params.k},Flat')
            index.train(np.array(vectors)) 
            index.nprobe = params.n_probe
            index.add_with_ids(np.array(vectors), np.array(ids))
                
            ## Store index to disk
            faiss.write_index(index, f'/tmp/{key_id}.ann')
            
            ## Upload index to storage
            storage.upload_file(f'/tmp/{key_id}.ann', params.storage_bucket, f'indexes/{params.dataset}/{params.implementation}/{params.num_index}/centroid_{key_id}.ann')

            ids = []
            vectors = []
                
            key_id += 1
            i += 1
    
    end = time.time()
    return end - start


def get_vectors_with_ids(args):
    
    start, size = args

    df = pd.read_csv(
        '/tmp/vectors.csv',
        header=None,
        skiprows=start,
        nrows=size
    )

    ids = df[0].tolist()
    vectors = [[float(x) for x in s.split() if x] for s in df[1]]
    del df
    return ids, vectors


def initialize_database(filename, params: SvlessVectorDBParams, fexec, num_workers):
    
    init = time.time()
    logging.info("Starting indexing")
    
    ## Distribute dataset across the different centroids
    vectors_key = params.storage_bucket + "/" + filename
        
    ## Generate an index for each centroid with the vectors assigned to it
    if params.implementation == "blocks":
        
        obj_chunk = 16
        n_blocks_per_function = int(params.num_index / obj_chunk)
        futures = fexec.map(generate_index_blocks, vectors_key, extra_args=[params, n_blocks_per_function], obj_chunk_number=obj_chunk, runtime_memory=params.index_mem)
        
        
    generate_index_time = fexec.get_result()
    lambda_invocation_indexing = [f.stats["worker_func_start_tstamp"] - f.stats["host_job_create_tstamp"] for f in futures]
    
    # Lithops plots
    end = time.time()
    
    timers = {}
    timers[f'generate_index_{params.implementation}'] = generate_index_time
    timers[f'generate_index_invocation_{params.implementation}'] = lambda_invocation_indexing
    timers[f'total_indexing_{params.implementation}'] = end - init
    
    return timers