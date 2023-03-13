
import csv
import argparse
import numpy as np

if __name__ == "__main__":
    pars = argparse.ArgumentParser()
    pars.add_argument('folder')
    args = pars.parse_args()
    filename = args.folder + '/colorLoss.csv'
    print("Filename: ", filename)
    avgs = []
    with open(filename) as csvfile:
        rows=csv.reader(csvfile)
        for row in rows:
            vals = np.array(row).astype(float)
            avgs += [np.mean(vals)]
    
    print(f"Color Loss Avg: {avgs[0]}")
    print(f"Depth Loss Avg: {avgs[1]}")