"""Fixed-budget matched linear/ReLU heads on frozen PANNs features."""
def train_matched_heads(
    x_train, y_train, x_clean, x_noise, y_val,
    output_dir, device="cuda"
):
    import csv
    import time
    from pathlib import Path

    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score

    torch.set_num_threads(2)
    device = torch.device(device)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Fit preprocessing exclusively on clean training embeddings.
    scaler = StandardScaler()
    xtr = scaler.fit_transform(
        np.asarray(x_train, dtype=np.float32)
    ).astype(np.float32)
    xcl = scaler.transform(
        np.asarray(x_clean, dtype=np.float32)
    ).astype(np.float32)
    xno = scaler.transform(
        np.asarray(x_noise, dtype=np.float32)
    ).astype(np.float32)

    ytr = np.asarray(y_train, dtype=np.int64)
    yval = np.asarray(y_val, dtype=np.int64)

    np.savez(
        out / "scaler.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
        var=scaler.var_,
    )

    dataset = TensorDataset(
        torch.from_numpy(xtr),
        torch.from_numpy(ytr),
    )
    clean_tensor = torch.from_numpy(xcl).to(device)
    noise_tensor = torch.from_numpy(xno).to(device)
    criterion = nn.CrossEntropyLoss()

    predictions = {}
    rows = []

    architectures = {
        "factorized_linear": nn.Identity,
        "mlp_relu": nn.ReLU,
    }

    def write_csv(path, records):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(records[0])
            )
            writer.writeheader()
            writer.writerows(records)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for architecture, activation in architectures.items():
        for seed in (101, 202, 303):

            # Reset initialization RNG for each architecture-seed pair.
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

            model = nn.Sequential(
                nn.Linear(2048, 49),
                activation(),
                nn.Linear(49, 50),
            ).to(device)

            parameters = sum(
                p.numel() for p in model.parameters()
            )

            # Independent generator reproduces minibatch ordering.
            generator = torch.Generator().manual_seed(seed)
            loader = DataLoader(
                dataset,
                batch_size=32,
                shuffle=True,
                generator=generator,
                num_workers=0,
                drop_last=False,
            )

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1e-3,
                weight_decay=1e-4,
            )

            history = []
            iterations = 0

            synchronize()
            start_time = time.perf_counter()

            # Fixed 100 epochs: no validation or checkpoint selection.
            for epoch in range(1, 101):
                model.train()
                loss_sum = 0.0

                for xb, yb in loader:
                    xb = xb.to(device)
                    yb = yb.to(device)

                    optimizer.zero_grad(set_to_none=True)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()

                    nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=5.0
                    )
                    optimizer.step()

                    loss_sum += loss.item() * len(yb)
                    iterations += 1

                history.append({
                    "epoch": epoch,
                    "train_ce_online": loss_sum / len(dataset),
                    "iterations": iterations,
                })

            synchronize()
            training_seconds = time.perf_counter() - start_time

            # Evaluate only the final-epoch weights.
            model.eval()
            with torch.inference_mode():

                # Exact final-model CE, not the online epoch average.
                total_ce = 0.0
                for i in range(0, len(dataset), 256):
                    xb = dataset.tensors[0][i:i+256].to(device)
                    yb = dataset.tensors[1][i:i+256].to(device)
                    loss = criterion(model(xb), yb)
                    total_ce += loss.item() * len(yb)

                train_ce = total_ce / len(dataset)

                clean_pred = (
                    model(clean_tensor)
                    .argmax(dim=1)
                    .cpu()
                    .numpy()
                )
                noise_pred = (
                    model(noise_tensor)
                    .argmax(dim=1)
                    .cpu()
                    .numpy()
                )

            print(f"Completed {architecture} seed {seed}: final train CE={train_ce:.6f}, seconds={training_seconds:.2f}", flush=True)
            prefix = f"{architecture}_seed{seed}"

            # Preserve all epoch losses and final trained weights.
            write_csv(out / f"{prefix}_history.csv", history)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "architecture": architecture,
                    "seed": seed,
                    "epoch": 100,
                },
                out / f"{prefix}.pt",
            )

            for condition, pred in (
                ("clean", clean_pred),
                ("noise10db", noise_pred),
            ):
                key = (architecture, seed, condition)
                predictions[key] = pred.copy()

                np.save(
                    out / f"{prefix}_{condition}_pred.npy",
                    pred,
                )

                rows.append({
                    "architecture": architecture,
                    "seed": seed,
                    "condition": condition,
                    "parameters": parameters,
                    "accuracy": float(
                        accuracy_score(yval, pred)
                    ),
                    "macroF1": float(
                        f1_score(
                            yval, pred,
                            labels=np.arange(50),
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    "trainCE": float(train_ce),
                    "iterations": iterations,
                    "time_s": training_seconds,
                })

    write_csv(out / "matched_heads_results.csv", rows)
    return predictions, rows
