import os
import argparse
import pandas as pd
import numpy as np
from mpi4py import MPI
from probability import log_probability
import h5py


def calc_probs(t, n0, n_max, Lambda, kr):
    # Initialize a list of zeros of length n_max for saving into a csv purposes
    probabilities = [0] * n_max
    # Calculate probabilities for each clone size from n0 to n_max
    calculated_probabilities = list(
        np.exp(
            [log_probability(t, n0, n, Lambda, kr) for n in np.arange(n0, n_max + 1, 1)]
        )
    )  # Note the probabilities calculated were log probabilities and so had to be turned into probabilities by exponentiating them and again converted to list.
    # Replace the elements from index n0 onwards with calculated probabilities
    probabilities[n0 - 1 :] = (
        calculated_probabilities  # Note that the least n0 can start from 1 and python list indexing starts at 0.
    )
    return probabilities


def calc_probs_parallel(args):
    t, n0, n_max, Lambda, kr = args
    result = calc_probs(t, n0, n_max, Lambda, kr)
    return result


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # print(rank)
    # print(comm)
    # print(comm.size)

    if not MPI.Is_initialized():
        if rank == 0:
            print(
                "MPI is not initialized. Make sure you have OpenMPI installed and properly configured."
            )
        MPI.Finalize()

    if rank == 0:
        print("MPI is initialized and running.")

    parser = argparse.ArgumentParser(description="Calculate probabilities.")
    parser.add_argument("--patient_id", type=str, required=True, help="patient id")
    parser.add_argument(
        "--o", type=str, default=os.getcwd(), help="Output filename: Give full path"
    )

    args = parser.parse_args()
    patient_id = args.patient_id
    output_filename = args.o

    ##################Prepare data###################
    patient_data = pd.read_csv(
        f"/home/gddaslab/mxp140/TCR_project/garfinkle/new_model/{patient_id}_merged_pre_and_post_vax_for_TCR_with_fold_change_gteq1.csv",
        sep=",",
    )

    params_df = pd.read_csv(
        "/home/gddaslab/mxp140/TCR_project/garfinkle/new_model/parameters.csv", sep=","
    )
    patient_params_df = params_df[params_df["subject"] == patient_id]
    Lambda_value = patient_params_df["parameter"].values[0]

    time_durations = {
        "subject1": 96,  # days
        "subject2": 68,  # days
        "subject3": 169,  # days
    }
    n_max_values = {
        "subject1": 5000,
        "subject2": 5000,
        "subject3": 10000,
    }
    t_value = time_durations.get(patient_id)
    n_max_value = n_max_values.get(patient_id)
    if t_value is None or n_max_value is None:
        raise ValueError(f"Either t or n_max is None or wrong: {t_value}")

    scaled_kr_values = patient_data["scaled_kr"].values
    n0_values = patient_data["cdr3_count_x"].values

    # Initialization
    if rank == 0:
        # Only the root process reads data and distributes to others
        chunk_size = size
        chunks_kr = [list(sublist) for sublist in np.array_split(scaled_kr_values, chunk_size)]  # type: ignore
        chunks_n0 = [list(sublist) for sublist in np.array_split(n0_values, chunk_size)]  # type: ignore
    else:
        chunks_kr = None
        chunks_n0 = None

    # Scatter chunked data each of comm.size length to processes in loops
    scaled_kr_values_scattered = comm.scatter(chunks_kr, root=0)  # type: ignore
    n0_values_scattered = comm.scatter(chunks_n0, root=0)  # type: ignore

    # Parallel execution
    results = []
    for i in range(len(scaled_kr_values_scattered)):
        local_results = calc_probs_parallel(
            (
                t_value,
                n0_values_scattered[i],
                n_max_value,
                Lambda_value,
                scaled_kr_values_scattered[i],
            )
        )
        results.append(local_results)

    # Gather results outside the loop
    results = comm.gather(results, root=0)

    # Root process prints results
    if rank == 0:
        # Note that local_results itself is a list so results will be some nested list. We will just flatten the gathered results list.
        results = [item for sublist in results for item in sublist]

        # Ensure the filename has the .h5 extension
        if not output_filename.endswith(".h5"):
            output_filename += ".h5"

        # Save as HDF5 file
        results_array = np.array(results)
        with h5py.File(output_filename, "w") as f:
            f.create_dataset("results", data=results_array)

        print("Results saved to HDF5 file:", output_filename)
        print("Full shape:", results_array.shape)

    # Finalize MPI
    MPI.Finalize()
