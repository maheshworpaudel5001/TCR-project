import os
import argparse
import pandas as pd
import numpy as np
from mpi4py import MPI
from model1.probability import probability
import h5py


def calc_probs(x1, x2, x1_dagger, x2_dagger, omega, kr, M):
    probabilities = [
        probability(x1, x2, x1_dagger, x2_dagger, omega, kr, M_value)
        for M_value in range(1, int(M) + 1)
    ]
    return probabilities


def calc_probs_parallel(args):
    x1, x2, x1_dagger, x2_dagger, omega, kr, M = args
    result = calc_probs(x1, x2, x1_dagger, x2_dagger, omega, kr, M)
    return result


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

    parser = argparse.ArgumentParser(description="Calculate probabilities.")
    parser.add_argument("--class_id", type=str, help="class id")
    parser.add_argument("--patient_id", type=str, required=True, help="patient id")
    parser.add_argument(
        "--o",
        type=str,
        default=os.path.join(os.getcwd(), "output_probabilities"),
        help="Output filename: Give full path. Otherwise defaults to output_probabilities.h5 at current working directory.",
    )

    args = parser.parse_args()
    class_id = args.class_id
    patient_id = args.patient_id
    output_filename = args.o

    ##################Prepare data###################
    full_data = pd.read_csv(
        f"/home/gddaslab/mxp140/TCR_project/data/full_{class_id}_data_v2.csv",
        sep=",",
        comment="#",
    )
    if "all" not in patient_id:
        patient_data = full_data[
            full_data["CDR3"].str.contains(patient_id)
        ].reset_index()
    else:
        patient_data = full_data

    params_df = pd.read_csv(
        "/home/gddaslab/mxp140/TCR_project/outputs/params_piecewise_omega_constrained_v2.csv",
        sep=",",
    )
    patient_params_df = params_df[params_df["Patient_ID"] == patient_id]
    x1_value = patient_params_df["x1"].values[0]
    x2_value = patient_params_df["x2"].values[0]
    x1_dagger_value = patient_params_df["x1_dagger"].values[0]
    x2_dagger_value = patient_params_df["x2_dagger"].values[0]
    omega_value = patient_params_df["w"].values[0]

    M_max_values = {
        "brmet008": 10000,
        "brmet009": 10000,
        "brmet010": 10000,
        "brmet018": 10000,
        "brmet019": 10000,
        "brmet025": 10000,
        "brmet027": 10000,
        "brmet028": 10000,
        "gbm032": 10000,
        "gbm052": 10000,
        "gbm055": 10000,
        "gbm056": 10000,
        "gbm059": 10000,
        "gbm062": 10000,
        "gbm063": 10000,
        "gbm064": 10000,
        "gbm070": 10000,
        "gbm074": 10000,
        "gbm079": 10000,
    }
    M_max_value = M_max_values.get(patient_id, 1000)

    scaled_kr_values = patient_data["scaled_kr"].values
    if len(scaled_kr_values) > 10000:
        splitted_scaled_kr_values = np.array_split(scaled_kr_values, 10)
    else:
        splitted_scaled_kr_values = [scaled_kr_values]  # put into list for consistency

    # Then process each chunk
    all_results = []
    for kr_chunk in splitted_scaled_kr_values:
        if rank == 0:
            chunk_kr = np.array_split(kr_chunk, size)
        else:
            chunk_kr = None

        chunk_kr_scattered = comm.scatter(chunk_kr, root=0)

        results = []
        for i, kr_value in enumerate(chunk_kr_scattered):
            local_results = calc_probs_parallel(
                (
                    x1_value,
                    x2_value,
                    x1_dagger_value,
                    x2_dagger_value,
                    omega_value,
                    kr_value,
                    M_max_value,
                )
            )
            results.append(local_results)
        all_results.extend(results)

    # Gather results outside the loop
    final_results = comm.gather(all_results, root=0)

    # Root process prints results
    if rank == 0:
        # Note that local_results itself is a list so results will be some nested list. We will just flatten the gathered results list.
        final_results = [item for sublist in final_results for item in sublist]

        # Ensure the filename has the .h5 extension
        if not output_filename.endswith(".h5"):
            output_filename += ".h5"

        # Save as HDF5 file
        results_array = np.array(final_results)
        with h5py.File(output_filename, "w") as f:
            f.create_dataset("results", data=results_array)

        print(
            "Results saved to HDF5 file:",
            output_filename,
            " with a key called 'results'.",
        )
        print("Full shape:", results_array.shape)

    # Finalize MPI
    MPI.Finalize()
