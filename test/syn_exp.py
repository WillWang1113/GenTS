from argparse import ArgumentParser

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import os
import gents.dataset
import gents.model

seed_everything(9)

dataset_names = gents.dataset.DATASET_NAMES
# dataset_names = ['SineND']
# model_names = ['FourierFlow']
model_names = gents.model.MODEL_NAMES
DEFAULT_ROOT_DIR = "/home/user/data2/GenTS_exp"
try:
    dataset_names.remove("Physionet")
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
        "--batch_size", type=int, default=64, help="Batch size for training."
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
        default=150,
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
    if univar:
        # model_names.remove("FIDE")  # only support univariate time series
        # model_names.remove("FourierFlow")  # only support univariate time series
        try:
            dataset_names.remove("Spiral2D")
        except:
            pass

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
                args["select_seq_dim"] = min(32, data_cls.D)

        for model_name in model_names:
            print("--" * 20)
            print(dataset_name, model_name)
            print("--" * 20)

            if model_name == "SDEGAN":
                args["add_coeffs"] = "linear"
            else:
                args["add_coeffs"] = "cubic_spline"

            dm = data_cls(**args)

            model_cls = getattr(gents.model, model_name)
            # filter out invalid condition
            if args["condition"] in model_cls.ALLOW_CONDITION:
                print(f"Testing model {model_name} on dataset {dataset_name}")
                model_args = dict(
                    seq_len=dm.seq_len, seq_dim=dm.seq_dim, condition=args["condition"]
                )

                model = model_cls(**model_args)

                trainer = Trainer(
                    max_epochs=max_epochs,
                    devices="auto" if model_name == "FourierFlow" else [gpu],
                    accelerator="cpu" if model_name == "FourierFlow" else "gpu",
                    callbacks=[
                        EarlyStopping(monitor="val_loss", patience=5, mode="min")
                    ],
                    default_root_dir=DEFAULT_ROOT_DIR,
                    # fast_dev_run=True,
                    enable_progress_bar=False
                )
                trainer.fit(model, dm)
                with open(
                    os.path.join(
                        DEFAULT_ROOT_DIR, f"{model_name}_{dataset_name}_ckptpth.txt"
                    ),
                    "w",
                ) as text_file:
                    text_file.write(trainer.checkpoint_callback.best_model_path)
                    text_file.close()
                # model = model_cls.load_from_checkpoint(
                #     trainer.checkpoint_callback.best_model_path
                # )

                # # model testing
                # model.eval()
                # model.to(f"cuda:{gpu}")
                # dm.setup("test")
                # y_pred = []
                # for test_batch in dm.test_dataloader():
                #     for k in test_batch:
                #         test_batch[k] = test_batch[k].to(f"cuda:{gpu}")
                #     samples = model.sample(
                #         n_sample=test_batch["seq"].shape[0],
                #         condition=test_batch.get("c", None),
                #         **test_batch,
                #     )
                #     y_pred.append(samples.detach())
                # y_pred = torch.cat(y_pred, dim=0).cpu()
                # print(y_pred.shape)
            # break
        if dataset_name == "SineND":
            args.pop("seq_dim")


if __name__ == "__main__":
    main()

