from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import os
import gents.dataset
from gents.evaluation.model_based import cfid
from gents.evaluation.model_based._ts2vec import initialize_ts2vec
from gents.evaluation.model_based.ds import discriminative_score
from gents.evaluation.model_based.ps import predictive_score
from gents.evaluation.model_free.distribution_distance import WassersteinDistances
import gents.model

seed_everything(9)

# dataset_names = ["Physionet"]
dataset_names = ["ECG"]
# dataset_names = ["Physionet", "Spiral2D", "ECG"]
# dataset_names = ['SineND']
# model_names = ["TimeVQVAE"]
model_names = gents.model.MODEL_NAMES
print("All available datasets: ", dataset_names)
print("All available models: ", model_names)

DEFAULT_ROOT_DIR = "/mnt/ExtraDisk/wcx/research/GenTS_cls"
try:
    # too large datasets
    # dataset_names.remove("Physionet")
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
        "--batch_size", type=int, default=64, help="Batch size for training."
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="class",
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
    seq_len = args.pop("seq_len")
    max_epochs = args.pop("max_epochs")
    univar = args.pop("univar")
    print(args)

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

    with open("syn_exp.txt", "a") as log_file:
        for dataset_name in dataset_names:
            data_cls = getattr(gents.dataset, dataset_name)

            args["select_seq_dim"] = None

            # if dataset_name == 'Physionet':
            #     args['agg_minutes'] = 120
            #     args.pop('seq_dim', None)
            #     class_num = 2
            # elif dataset_name == 'ECG':
            #     class_num = 5
            # elif dataset_name == 'Spiral2D':
            #     class_num = 2

            for model_name in model_names:
                print("==" * 20)
                print(dataset_name, model_name)
                ckpt_path = os.path.join(
                    DEFAULT_ROOT_DIR, f"{model_name}_{dataset_name}_ckptpth.txt"
                )
                print("Checkpoint path:", ckpt_path)

                if not os.path.exists(ckpt_path):
                    print(
                        f"Model {model_name} on dataset {dataset_name} didn't trained. Skipping."
                    )
                    continue
                args["seq_len"] = seq_len
                dm = data_cls(**args)
                print(args)

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
                    # model_args = dict(
                    #     seq_len=dm.seq_len,
                    #     seq_dim=dm.seq_dim,
                    #     condition=args["condition"],
                    #     class_num=dm.n_classes,
                    # )
                    with open(ckpt_path, "r") as f:
                        model_ckpt = f.read().strip()

                    # model = model_cls(**model_args)
                    model = model_cls.load_from_checkpoint(model_ckpt)


                    # model testing
                    model.eval()
                    model.to(f"cuda:{gpu}")
                    dm.setup("fit")
                    # y_pred = []
                    y_pred, y_real, y_real_label = [], [], []
                    y_real = dm.train_ds.data.to(f"cuda:{gpu}")
                    y_real_label = dm.train_ds.class_labels.to(f"cuda:{gpu}")

                    y_pred = model.sample(
                        n_sample=y_real.shape[0],
                        condition=y_real_label.int(),
                    )
                    # test_batch = torch.concat([batch] for batch in dm.train_dataloader())
                    # for test_batch in dm.train_dataloader():
                    #     y_real.append(test_batch["seq"])
                    #     y_real_label.append(test_batch.get("c", None))

                    #     for k in test_batch:
                    #         test_batch[k] = test_batch[k].to(f"cuda:{gpu}")
                    #     samples = model.sample(
                    #         n_sample=test_batch["seq"].shape[0],
                    #         condition=test_batch.get("c", None),
                    #         **test_batch,
                    #     )
                    #     y_pred.append(samples.detach())

                    # y_pred = torch.cat(y_pred, dim=0).cpu()
                    # y_real = torch.cat(y_real, dim=0).cpu()
                    # y_real_label = torch.cat(y_real_label).cpu()
                    print(y_pred.shape)
                    print(y_real.shape)
                    print(y_real_label.shape)

                    torch.save(
                        y_pred.cpu(),
                        os.path.join(
                            DEFAULT_ROOT_DIR,
                            f"{model_name}_{dataset_name}_y_pred.pt",
                        ),
                    )
                    # if not os.path.exists(
                    #     os.path.join(
                    #         DEFAULT_ROOT_DIR,
                    #         f"{dataset_name}_y_real.pt",
                    #     )
                    # ):
                    torch.save(
                        y_real.cpu(),
                        os.path.join(
                            DEFAULT_ROOT_DIR,
                            f"{dataset_name}_y_real.pt",
                        ),
                    )

                    
                    torch.save(
                        y_real_label.cpu(),
                        os.path.join(
                            DEFAULT_ROOT_DIR,
                            f"{dataset_name}_y_real_label.pt",
                        ),
                    )

                    cfid_score = cfid.context_fid(
                        # train_data=train_data.numpy(),
                        ori_data=y_real.cpu().numpy(),
                        gen_data=y_pred.cpu().numpy(),
                        device=f"cuda:{gpu}",
                        ts2vec_path=os.path.join(
                            DEFAULT_ROOT_DIR, "ts2vec", f"{dataset_name}.pt"
                        ),
                    )
                    print("Done cfid")
                    w_dist = WassersteinDistances(
                        y_real.cpu().flatten(1).numpy(), y_pred.cpu().flatten(1).numpy()
                    )
                    wd_score = w_dist.sliced_distances(500).mean()
                    print("Done WD")

                    p_score = predictive_score(
                        y_real.cpu().numpy(),
                        y_pred.cpu().numpy(),
                        device=f"cuda:{gpu}",
                    )
                    print("Done ps")
                    d_score = discriminative_score(
                        y_real.cpu().numpy(),
                        y_pred.cpu().numpy(),
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
                            f"{model_name}_{dataset_name}_metrics_insample_new.csv",
                        ),
                        index=False,
                    )
                # print("--" * 20)

            if dataset_name == "SineND":
                args.pop("seq_dim")
            if dataset_name == "Physionet":
                args["seq_len"] = seq_len


if __name__ == "__main__":
    main()
