import numpy as np
import random
import os

TTL_SAMPLE = 10

def get_subset(idx_start=0, ttl_sample=20_000, sub_size=1.0):
    ret = random.sample(range(idx_start, idx_start+ttl_sample), int(sub_size*ttl_sample))
    return ret
    pass

def main():
    
    percentage = 0.2

    subset_idx_list = []
    for mod_i in range(0, 10):
        idx_start = mod_i*TTL_SAMPLE
        subset_idx = get_subset(idx_start=idx_start, ttl_sample=TTL_SAMPLE, sub_size=percentage)
        subset_idx_list += subset_idx
    print(f"Total idx: {subset_idx_list}")
    pass

if __name__ == '__main__':
    main()