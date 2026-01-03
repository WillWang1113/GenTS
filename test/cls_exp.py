from argparse import ArgumentParser

import numpy as np
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import os
import gents.dataset
import gents.model

seed_everything(9)

dataset_names = ['Physionet', 'ECG', 'Spiral2D']
# dataset_names = ['SineND']
model_names = ['TimeVQVAE']
# model_names = gents.model.MODEL_NAMES
print("All available datasets: ", dataset_names)
print("All available models: ", model_names)

DEFAULT_ROOT_DIR = "/mnt/ExtraDisk/wcx/research/GenTS_cls_new"
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
        default='class',
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
        default=100,
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
            #     # class_num = 2
            # elif dataset_name == 'ECG':
            #     # class_num = 5
            #     pass
            # elif dataset_name == 'Spiral2D':
            #     # class_num = 2
            #     pass
                    

            for model_name in model_names:
                print("==" * 20)
                print(dataset_name, model_name)

                

                if os.path.exists(
                    os.path.join(
                        DEFAULT_ROOT_DIR, f"{model_name}_{dataset_name}_ckptpth.txt"
                    )
                ):
                    print(
                        f"Model {model_name} on dataset {dataset_name} already trained. Skipping."
                    )
                    continue

                dm = data_cls(**args)

                model_cls = getattr(gents.model, model_name)
                # filter out invalid condition
                if args["condition"] in model_cls.ALLOW_CONDITION:
                    print(f"Testing model {model_name} on dataset {dataset_name}")
                    model_args = dict(
                        seq_len=dm.seq_len,
                        seq_dim=dm.seq_dim,
                        condition=args["condition"],
                        class_num=dm.n_classes
                    )
                    if model_name == "VanillaDDPM":
                        # model_args['d_model'] = 32
                        # model_args['n_layers'] = 1
                        model_args['patch_size'] = 24
                    
                    if model_name == "TimeVQVAE":
                        model_args['cfg_scale'] = 1.0
                        
                        
                    model = model_cls(**model_args)

                    trainer = Trainer(
                        max_epochs=max_epochs,
                        devices=[gpu],
                        accelerator="gpu",
                        callbacks=[
                            EarlyStopping(monitor="val_loss", patience=10, mode="min")
                        ],
                        default_root_dir=DEFAULT_ROOT_DIR,
                        min_epochs=max_epochs,
                        # min_epochs=max_epochs if model_name == 'TimeVQVAE' else None,
                        # fast_dev_run=True,
                        enable_progress_bar=False,
                        enable_model_summary=False
                    )
                    try:
                        trainer.fit(model, dm)
                    except Exception as e:
                        print(
                            f"Error training model {model_name} on dataset {dataset_name}: {e}"
                        )
                        log_file.write(
                            f"Error training model {model_name} on dataset {dataset_name}: {e}\n"
                        )
                        continue
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
                # print("--" * 20)
                    
            if dataset_name == "SineND":
                args.pop("seq_dim")
            if dataset_name == 'Physionet':
                args['seq_len'] = seq_len
                
            


if __name__ == "__main__":
    main()
