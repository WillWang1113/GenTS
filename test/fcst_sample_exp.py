import os
from argparse import ArgumentParser

import numpy as np
import torch
from lightning import seed_everything

import gents.dataset
import gents.model
from gents.evaluation.model_free.errors import crps

seed_everything(9)

# dataset_names = gents.dataset.DATASET_NAMES
dataset_names = ['SineND']
model_names = ['VanillaVAE']
# model_names = gents.model.MODEL_NAMES
print("All available datasets: ", dataset_names)
print("All available models: ", model_names)

DEFAULT_ROOT_DIR = "/home/user/data2/GenTS_fcst_exp"
try:
    # too large datasets
    dataset_names.remove("Physionet")
    dataset_names.remove("ETTm1")
    dataset_names.remove("ETTm2")

    # too slow model
    # model_names.remove("GTGAN")
    # model_names.remove("LatentSDE")
except:
    pass


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--seq_len", type=int, default=24, help="Length of the time series sequences."
    )
    parser.add_argument(
        "--univar", action="store_true", help="Length of the time series sequences."
    )
    parser.add_argument(
        "--inference_batch_size", type=int, default=512, help="Batch size for training."
    )
    parser.add_argument(
        "--n_sample", type=int, default=50, help="Number of samples for inference."
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
        default=None,
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
    max_epochs = args.pop("max_epochs")
    univar = args.pop("univar")
    if args["condition"] == "predict":
        args["obs_len"] = args["seq_len"]
    elif args["condition"] == "impute":
        args["missing_rate"] = 0.2
    if univar:
        # model_names.remove("FIDE")  # only support univariate time series
        # model_names.remove("FourierFlow")  # only support univariate time series
        try:
            dataset_names.remove("Spiral2D")
        except:
            pass

    with open("syn_exp.txt", "a") as log_file:
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
                        DEFAULT_ROOT_DIR,
                        f"{model_name}_{dataset_name}_metrics.csv",
                    )
                ):
                    print(
                        f"Model {model_name} on dataset {dataset_name} already tested. Skipping."
                    )
                    continue
                
                print(args)
                dm = data_cls(**args)

                model_cls = getattr(gents.model, model_name)
                # filter out invalid condition
                if args["condition"] in model_cls.ALLOW_CONDITION:
                    print(f"Testing model {model_name} on dataset {dataset_name}")
                    model_args = dict(
                        seq_len=dm.seq_len,
                        seq_dim=dm.seq_dim,
                        condition=args["condition"],
                    )
                    if args["condition"] == "predict":
                        model_args["obs_len"] = args["obs_len"]
                    elif args["condition"] == "impute":
                        model_args["missing_rate"] = args["missing_rate"]

                    if model_name == "ImagenTime" and args["condition"] == "predict":
                        model_args["delay"] = 12
                        model_args["embedding"] = 16
                    else:
                        pass

                    # model = model_cls(**model_args)
                    with open(ckpt_path, "r") as f:
                        model_ckpt = f.read().strip()

                    model = model_cls.load_from_checkpoint(model_ckpt, strict=False, map_location="cpu")

                    # model testing
                    model.eval()
                    # model.to(f"cuda:{gpu}")
                    dm.setup("test")
                    y_pred, y_real = [], []
                    avg_metrics = {"mse": 0.0, "crps": 0.0}
                    for test_batch in dm.test_dataloader():
                        # for k in test_batch:
                            # test_batch[k] = test_batch[k].to(f"cuda:{gpu}")
                        # while True:
                        samples = model.sample(
                            n_sample=args["n_sample"],
                            condition=test_batch.get("c", None),
                            **test_batch,
                        )

                        if torch.isnan(samples).any():
                            print("has nan in samples, try again")
                            samples = torch.nan_to_num(samples)

                        y_pred.append(samples)
                        y_real.append(test_batch["seq"].cpu())

                    y_pred = torch.concat(y_pred).detach().cpu()
                    y_real = torch.concat(y_real).cpu()[:, -args['seq_len']:, ...]
                    print(y_real.shape)
                    print(y_pred.shape)
                    avg_metrics["mse"] = torch.nn.functional.mse_loss(
                        y_real, y_pred.mean(dim=-1)
                    ).item()
                    avg_metrics["crps"] = crps(y_real.numpy(), y_pred.numpy())

                    # avg_metrics["mse"] /= len(dm.test_dataloader())
                    # avg_metrics["crps"] /= len(dm.test_dataloader())
                    # y_pred = torch.cat(y_pred, dim=0).cpu()
                    # print(y_pred.shape)
                # print("--" * 20)
                print(avg_metrics)

                # pd.DataFrame(avg_metrics, index=[0]).to_csv(
                #     os.path.join(
                #         DEFAULT_ROOT_DIR,
                #         f"{model_name}_{dataset_name}_metrics.csv",
                #     ),
                #     index=False,
                # )

                # break
            if dataset_name == "SineND":
                args.pop("seq_dim")


if __name__ == "__main__":
    main()
