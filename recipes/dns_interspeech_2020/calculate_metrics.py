import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))  # add /path/to/FullSubNet
print(sys.path)

from inspect import getmembers, isfunction
from pathlib import Path

import librosa
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

import audio_zen.metrics as metrics
from audio_zen.utils import prepare_empty_dir


def load_wav_paths_from_scp(scp_path, to_abs=True):
    wav_paths = [line.rstrip('\n') for line in open(os.path.abspath(os.path.expanduser(scp_path)), "r")]
    if to_abs:
        tmp = []
        for path in wav_paths:
            tmp.append(os.path.abspath(os.path.expanduser(path)))
        wav_paths = tmp
    return wav_paths


def shrink_multi_channel_path(
        full_dataset_list: list,
        num_channels: int
) -> list:
    """

    Args:
        full_dataset_list: [
            028000010_room1_rev_RT600.06_mic1_micpos1.5p0.5p1.93_srcpos0.46077p1.1p1.68_langle180_angle150_ds1.2_mic1.wav
            ...
            028000010_room1_rev_RT600.06_mic1_micpos1.5p0.5p1.93_srcpos0.46077p1.1p1.68_langle180_angle150_ds1.2_mic2.wav
        ]
        num_channels:

    Returns:

    """
    assert len(full_dataset_list) % num_channels == 0, "Num error"

    shrunk_dataset_list = []
    for index in range(0, len(full_dataset_list), num_channels):
        full_path = full_dataset_list[index]
        shrunk_path = f"{'_'.join(full_path.split('_')[:-1])}.wav"
        shrunk_dataset_list.append(shrunk_path)

    assert len(shrunk_dataset_list) == len(full_dataset_list) // num_channels
    return shrunk_dataset_list


def get_basename(path):
    return os.path.splitext(os.path.basename(path))[0]


def pre_processing(est, ref, specific_dataset=None):
    ref = Path(ref).expanduser().absolute()
    est = Path(est).expanduser().absolute()

    if ref.is_dir():
        reference_wav_paths = librosa.util.find_files(ref.as_posix(), ext="wav")
    else:
        reference_wav_paths = load_wav_paths_from_scp(ref.as_posix())

    if est.is_dir():
        estimated_wav_paths = librosa.util.find_files(est.as_posix(), ext="wav")
    else:
        estimated_wav_paths = load_wav_paths_from_scp(est.as_posix())

    if not specific_dataset:
        # By default, the two lists should have a one-to-one correspondence
        check_two_aligned_list(reference_wav_paths, estimated_wav_paths)
    else:
        #For different data sets, manual alignment is performed to ensure one-to-one correspondence between the two lists
        reordered_estimated_wav_paths = []
        if specific_dataset == "dns_1":
            #Rearrange estimated_wav_paths according to the suffix of the file in reference_wav_paths
            # Extract suffix
            for ref_path in reference_wav_paths:
                for est_path in estimated_wav_paths:
                    est_basename = get_basename(est_path)
                    if "clean_" + "_".join(est_basename.split("_")[-2:]) == get_basename(ref_path):
                        reordered_estimated_wav_paths.append(est_path)
        elif specific_dataset == "dns_2":
            for ref_path in reference_wav_paths:
                for est_path in estimated_wav_paths:
                    # synthetic_french_acejour_orleans_sb_64kb-01_jbq2HJt9QXw_snr14_tl-26_fileid_47
                    # synthetic_clean_fileid_47
                    est_basename = get_basename(est_path)
                    file_id = est_basename.split('_')[-1]
                    if f"synthetic_clean_fileid_{file_id}" == get_basename(ref_path):
                        reordered_estimated_wav_paths.append(est_path)
        elif specific_dataset == "maxhub_noisy":
            # Reference_channel = 0
            # Find the corresponding clean voice
            reference_channel = 0
            print(f"Found #files: {len(reference_wav_paths)}")
            for est_path in estimated_wav_paths:
                # MC0604W0154_room4_rev_RT600.1_mic1_micpos1.5p0.5p1.84_srcpos4.507p1.5945p1.3_langle180_angle20_ds3.2_kesou_kesou_mic1.wav
                est_basename = get_basename(est_path)  # Noisy
                for ref_path in reference_wav_paths:
                    ref_basename = get_basename(ref_path)

        else:
            raise NotImplementedError(f"Not supported specific dataset {specific_dataset}.")
        estimated_wav_paths = reordered_estimated_wav_paths

    return reference_wav_paths, estimated_wav_paths



def check_two_aligned_list(a, b):
    assert len(a) == len(b), "The lengths in the two lists are not equal."
    for z, (i, j) in enumerate(zip(a, b), start=1):
        assert get_basename(i) == get_basename(j), f"There are different file names in the two lists, and the number of lines is: {z}" \
                                                   f"\n\t {i}" \
                                                   f"\n\t{j}"


def compute_metric(reference_wav_paths, estimated_wav_paths, sr, metric_type="SI_SDR"):
    metrics_dict = {o[0]: o[1] for o in getmembers(metrics) if isfunction(o[1])}
    assert metric_type in metrics_dict, f"Unsupported evaluation index： {metric_type}"
    metric_function = metrics_dict[metric_type]
    def calculate_metric(ref_wav_path, est_wav_path):
        ref_wav, _ = librosa.load(ref_wav_path, sr=sr)
        est_wav, _ = librosa.load(est_wav_path, sr=sr, mono=False)
        if est_wav.ndim > 1:
            est_wav = est_wav[0]

        basename = get_basename(ref_wav_path)

        ref_wav_len = len(ref_wav)
        est_wav_len = len(est_wav)

        if ref_wav_len != est_wav_len:
            print(f"[Warning] ref {ref_wav_len} and est {est_wav_len} are not in the same length")
            pass

        return basename, metric_function(ref_wav[:len(est_wav)], est_wav)

    metrics_result_store = Parallel(n_jobs=40)(
        delayed(calculate_metric)(ref, est) for ref, est in tqdm(zip(reference_wav_paths, estimated_wav_paths))
    )
    return metrics_result_store


def main(args):
    sr = args.sr
    metric_types = args.metric_types
    export_dir = args.export_dir
    specific_dataset = args.specific_dataset.lower()

    # Obtain all wav samples through the specified scp file or directory
    reference_wav_paths, estimated_wav_paths = pre_processing(args.estimated, args.reference, specific_dataset)

    if export_dir:
        export_dir = Path(export_dir).expanduser().absolute()
        prepare_empty_dir([export_dir])

    print(f"=== {args.estimated} === {args.reference} ===")
    for metric_type in metric_types.split(","):
        metrics_result_store = compute_metric(reference_wav_paths, estimated_wav_paths, sr, metric_type=metric_type)

        # Print result
        metric_value = np.mean(list(zip(*metrics_result_store))[1]) if len(metrics_result_store) != 0 else 0
        print(f"{metric_type}: {metric_value}")

        # Export result
        if export_dir:
            import tablib

            export_path = export_dir / f"{metric_type}.xlsx"
            print(f"Export result to {export_path}")

            headers = ("Speech", f"{metric_type}")
            metric_seq = [[basename, metric_value] for basename, metric_value in metrics_result_store]
            data = tablib.Dataset(*metric_seq, headers=headers)
            with open(export_path.as_posix(), "wb") as f:
                f.write(data.export("xlsx"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
    description="Enter two directories or lists to calculate the mean value of various evaluation indicators",
        epilog="python calculate_metrics.py -E 'est_dir' -R 'ref_dir' -M SI_SDR,STOI,WB_PESQ,NB_PESQ,SSNR,LSD,SRMR"
    )
    parser.add_argument("-R", "--reference", required=True, type=str, help="")
    parser.add_argument("-E", "--estimated", required=True, type=str, help="")
    parser.add_argument("-M", "--metric_types", required=True, type=str, help="Which evaluation index should be consistent with the content in util.metrics.")
    parser.add_argument("--sr", type=int, default=16000, help="Sampling Rate")
    parser.add_argument("-D", "--export_dir", type=str, default="", help="")
    parser.add_argument("--limit", type=int, default=None, help="[Under development] The maximum number of files read from the list.")
    parser.add_argument("--offset", type=int, default=0, help="[Under development] Start reading the file from the specified position in the list.")
    parser.add_argument("-S", "--specific_dataset", type=str, default="", help="Specify the data set type, e.g. DNS_1, DNS_2, both upper and lower case")
    args = parser.parse_args()
    main(args)

    """
    TODO
    1. How to calculate when the voice is multi-channel
    2. Support register, by default, all voices in register should be counted
    """
