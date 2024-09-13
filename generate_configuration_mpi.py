import os
import h5py
import json
import argparse
import numpy as np
from mpi4py import MPI
from collections import defaultdict


def process_row(row, num_columns):
    selected_values = []
    for _ in range(50):
        selected_values.extend(
            np.random.choice(np.arange(1, num_columns + 1), size=1000, p=row / sum(row))
        )
    return selected_values


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if not MPI.Is_initialized():
        if rank == 0:
            print(
                "MPI is not initialized. Make sure you have OpenMPI installed and properly configured."
            )
        MPI.Finalize()

    if rank == 0:
        print("MPI is initialized and running.")

    parser = argparse.ArgumentParser(
        description="Generate configurations from probability data."
    )
    parser.add_argument("--i", type=str, required=True, help="Input file in .h5 format")
    parser.add_argument(
        "--o",
        type=str,
        default=os.path.join(os.getcwd(), "output_configurations"),
        help="Output filename: Give full path. Otherwise defaults to output_configurations.json at current working directory.",
    )

    args = parser.parse_args()
    input_file, output_file = args.i, args.o

    if rank == 0:
        # Load the .h5 file
        with h5py.File(input_file, "r") as f:
            probability_array = f["result"][:]
            # probability_array =/
        num_columns = probability_array.shape[1]
        # Split the data among the available processes
        data_split = np.array_split(probability_array, size)  # splits row wise
    else:
        data_split = None
        num_columns = None

    # Broadcast the number of columns to all processes
    num_columns = comm.bcast(num_columns, root=0)

    # Scatter the data to all processes
    data_chunk = comm.scatter(data_split, root=0)

    # Each process processes its chunk of data
    local_counts = defaultdict(int)
    for row in data_chunk:
        selected_values = process_row(row, num_columns)
        for value in selected_values:
            local_counts[value] += 1

    # Gather the local counts from all processes
    all_counts = comm.gather(local_counts, root=0)

    if rank == 0:
        # Combine the counts from all processes
        global_counts = defaultdict(int)
        for counts in all_counts:
            for key, value in counts.items():
                global_counts[int(key)] += value  # Convert keys to standard Python int

        # Convert the counts dictionary to a regular dictionary and sort by keys
        global_counts = dict(sorted(global_counts.items()))

        # Ensure the output filename has the .json extension
        if not output_file.endswith(".json"):
            output_file += ".json"
        # Save the counts dictionary as a json file
        with open(output_file, "w") as f:
            json.dump(global_counts, f)

        print("Results saved to json file:", output_file)

    # Finalize MPI
    MPI.Finalize()
