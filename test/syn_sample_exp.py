import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from lightning import seed_everything

import gents.dataset
import gents.model
from gents.evaluation.model_based import cfid
from gents.evaluation.model_based._ts2vec import initialize_ts2vec
from gents.evaluation.model_based.ds import discriminative_score
from gents.evaluation.model_based.ps import predictive_score
from gents.evaluation.model_free.distribution_distance import WassersteinDistances

seed_everything(9)

# dataset_names = gents.dataset.DATASET_NAMES[7:]
dataset_names = ["Stocks"]
model_names = ["VanillaVAE"]
# model_names = gents.model.MODEL_NAMES 


DEFAULT_ROOT_DIR = "/mnt/ExtraDisk/wcx/research/GenTS_multivar_syn"
try:
    # too large datasets
    dataset_names.remove("Physionet")
    dataset_names.remove("ETTm1")
    dataset_names.remove("ETTm2")

    # model_names.remove("PSAGAN")
    model_names.remove("FourierDiffusionMLP")
    model_names.remove("FourierDiffusionLSTM")

    # too slow model
    # model_names.remove("GTGAN")
    # model_names.remove("LatentSDE")
except:
    pass

print("All available datasets: ", dataset_names)
print("All available models: ", model_names)

def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--seq_len", type=int, default=24, help="Length of the time series sequences."
    )
    parser.add_argument(
        "--univar", action="store_true", help="Length of the time series sequences."
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training."
    )
    parser.add_argument(
        "--inference_batch_size",
        type=int,
        default=1024,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default=None,
        choices=[None, "predict", "impute", "class"],
        help="Condition type (e.g., [None, 'predict', 'impute', 'class']).",
    )
    parser.add_argument(
        "--add_coeffs",
        type=str,
        default="cubic_spline",
        choices=[None, "linear", "cubic_spline"],
        help="Type of coefficients to add (e.g., 'cubic_spline').",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=300,
        help="Maximum number of epochs to train the model.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index to use",
    )
    args = parser.parse_args()
    return args


def main():
    print("--" * 20)
    print("Available datasets: ", dataset_names)
    print("Available synthetic models: ", model_names)
    print("--" * 20)

    args = parse_args()
    args = vars(args)
    args["data_dir"] = os.path.join(DEFAULT_ROOT_DIR, "data")
    gpu = args.pop("gpu")
    # max_epochs = args.pop("max_epochs")
    univar = args.pop("univar")
    if univar:
        # model_names.remove("FIDE")  # only support univariate time series
        # model_names.remove("FourierFlow")  # only support univariate time series
        try:
            dataset_names.remove("Spiral2D")
        except:
            pass
    else:
        try:
            model_names.remove("FIDE")
            model_names.remove("FourierFlow")
        except:
            pass

    with open("syn_sample_exp.txt", "a") as log_file:
        for dataset_name in dataset_names:
            data_cls = getattr(gents.dataset, dataset_name)

            if univar:
                if dataset_name in ["SineND"]:
                    args["seq_dim"] = 1
                else:
                    args["select_seq_dim"] = [
                        0
                    ]  # only use the first dimension for multivariate time series
            else:
                if dataset_name in ["SineND"]:
                    # Default seq_dim=5 for SineND in TimeGAN
                    args["seq_dim"] = 5
                else:
                    # Limit to seq_dim=32 for multivariate time series,
                    # Limit Electricity and Traffic, which have 321 and 862 dimensions respectively
                    total_dim = min(16, data_cls.D)
                    args["select_seq_dim"] = np.random.permutation(total_dim).tolist()

            for model_name in model_names:
                print("==" * 20)
                print(dataset_name, model_name)
                ckpt_path = os.path.join(
                    DEFAULT_ROOT_DIR, f"{model_name}_{dataset_name}_ckptpth.txt"
                )
                print("Checkpoint path:", ckpt_path)

                if model_name == "SDEGAN":
                    args["add_coeffs"] = "linear"
                elif model_name == "GTGAN":
                    args["add_coeffs"] = "cubic_spline"
                else:
                    args["add_coeffs"] = None

                if not os.path.exists(ckpt_path):
                    print(
                        f"Model {model_name} on dataset {dataset_name} didn't trained. Skipping."
                    )
                    continue

                if os.path.exists(
                    os.path.join(
                        DEFAULT_ROOT_DIR, f"{model_name}_{dataset_name}_metrics_insample.csv"
                    )
                ):
                    print(
                        f"Model {model_name} on dataset {dataset_name} already tested. Skipping."
                    )
                    continue

                dm = data_cls(**args)

                # pretrain ts2vec
                dm.setup("fit")
                train_data = []
                for train_batch in dm.train_dataloader():
                    train_data.append(train_batch["seq"])
                train_data = torch.cat(train_data, dim=0)

                initialize_ts2vec(
                    train_data.shape[-1],
                    train_data.numpy(),
                    device=f"cuda:{gpu}",
                    ts2vec_path=os.path.join(
                        DEFAULT_ROOT_DIR, "ts2vec", f"{dataset_name}.pt"
                    ),
                )

                model_cls = getattr(gents.model, model_name)
                # filter out invalid condition
                if args["condition"] in model_cls.ALLOW_CONDITION:
                    print(f"Testing model {model_name} on dataset {dataset_name}")
                    with open(ckpt_path, "r") as f:
                        model_ckpt = f.read().strip()
                    try:
                        model = model_cls.load_from_checkpoint(model_ckpt)
                    except:
                        try:
                            model = model_cls.load_from_checkpoint(
                                model_ckpt, strict=False
                            )
                        except Exception as e:
                            print(
                                f"Error sampling model {model_name} on dataset {dataset_name}: {e}"
                            )
                            log_file.write(
                                f"Error sampling model {model_name} on dataset {dataset_name}: {e}\n"
                            )
                            continue

                    model = model_cls.load_from_checkpoint(model_ckpt, strict=False)

                    # model testing
                    model.eval()
                    model.to(f"cuda:{gpu}")
                    dm.setup("test")
                    y_pred, y_real = [], []
                    all_num_samples = 0
                    # Insample performance
                    for test_batch in dm.train_dataloader():
                    # for test_batch in dm.test_dataloader():
                        y_real.append(test_batch["seq"])
                        for k in test_batch:
                            test_batch[k] = test_batch[k].to(f"cuda:{gpu}")
                        # try:
                        samples = model.sample(
                            n_sample=test_batch["seq"].shape[0],
                            condition=None,
                            **test_batch,
                        )

                        if model_name == "PSAGAN":
                            samples = torch.nn.functional.interpolate(
                                samples.permute(0, 2, 1),
                                size=args["seq_len"],
                                mode="linear",
                            ).permute(0, 2, 1)

                            # interp_samples = torch.stack(interp_samples, dim=-1)
                            # samples = interp_samples
                        # except Exception as e:
                        #     print(
                        #         f"Error sampling model {model_name} on dataset {dataset_name}: {e}"
                        #     )
                        #     log_file.write(
                        #         f"Error sampling model {model_name} on dataset {dataset_name}: {e}\n"
                        #     )
                        #     break
                        if torch.isnan(samples).any():
                            print("has nan in samples, try again")
                            samples = torch.nan_to_num(samples)
                        y_pred.append(samples.detach())
                        all_num_samples += samples.shape[0]
                        if all_num_samples >= 10000:
                            break
                    # print(y_pred)
                    y_pred = torch.cat(y_pred, dim=0).cpu()
                    y_real = torch.cat(y_real, dim=0).cpu()
                    print(y_pred.shape)
                    print(y_real.shape)
                    assert y_pred.shape == y_real.shape
                    print("Generated samples shape:", y_pred.shape)
                    torch.save(
                        y_pred,
                        os.path.join(
                            DEFAULT_ROOT_DIR,
                            f"{model_name}_{dataset_name}_y_pred.pt",
                        ),
                    )
                    if not os.path.exists(os.path.join(
                        DEFAULT_ROOT_DIR,
                        f"{dataset_name}_y_real.pt",
                    )):
                        torch.save(y_real, os.path.join(
                            DEFAULT_ROOT_DIR,
                            f"{dataset_name}_y_real.pt",
                        ))
                    cfid_score = cfid.context_fid(
                        # train_data=train_data.numpy(),
                        ori_data=y_real.numpy(),
                        gen_data=y_pred.numpy(),
                        device=f"cuda:{gpu}",
                        ts2vec_path=os.path.join(
                            DEFAULT_ROOT_DIR, "ts2vec", f"{dataset_name}.pt"
                        ),
                    )
                    print("Done cfid")
                    w_dist = WassersteinDistances(
                        y_real.flatten(1).numpy(), y_pred.flatten(1).numpy()
                    )
                    wd_score = w_dist.sliced_distances(500).mean()
                    print("Done WD")

                    p_score = predictive_score(
                        y_real.numpy(),
                        y_pred.numpy(),
                        device=f"cuda:{gpu}",
                    )
                    print("Done ps")
                    d_score = discriminative_score(
                        y_real.numpy(),
                        y_pred.numpy(),
                        device=f"cuda:{gpu}",
                    )
                    print("Done ds")
                    avg_metrics = {
                        "cfid": cfid_score,
                        "wd": wd_score,
                        "ps": p_score,
                        "ds": d_score,
                    }
                    print(avg_metrics)
                    pd.DataFrame(avg_metrics, index=[0]).to_csv(
                        os.path.join(
                            DEFAULT_ROOT_DIR,
                            f"{model_name}_{dataset_name}_metrics_insample.csv",
                        ),
                        index=False,
                    )

                print("--" * 20)

                # break
            if dataset_name == "SineND":
                args.pop("seq_dim")


if __name__ == "__main__":
    main()
