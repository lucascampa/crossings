"""FT-Transformer binary classifier with sklearn-style fit and predict_proba.

Wraps ``rtdl_revisiting_models.FTTransformer`` (Gorishniy et al. 2021) so it can
be tuned by ``collab_sol.tune_model`` like any other model. It expects the
matrix layout produced by ``ft_transformer.preprocess.make_preprocessor``:
continuous columns first, then ``n_cat_features`` integer category codes where
-1 means "unseen at fit time".
"""

from __future__ import annotations

import copy
from typing import Any, Self

import numpy as np
import torch
from numpy.typing import NDArray
from rtdl_revisiting_models import FTTransformer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from torch import nn


class FTTransformerClassifier:
    """CUDA-capable FT-Transformer for rare-event binary risk scoring."""

    def __init__(
        self,
        *,
        n_cat_features: int,
        n_blocks: int = 3,
        d_block: int = 192,
        attention_n_heads: int = 8,
        attention_dropout: float = 0.2,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        pos_weight: float | None = None,
        batch_size: int = 1024,
        max_epochs: int = 50,
        patience: int = 5,
        fixed_epochs: int | None = None,
        val_fraction: float = 0.1,
        random_state: int = 42,
    ) -> None:
        self.n_cat_features = n_cat_features
        self.n_blocks = n_blocks
        self.d_block = d_block
        self.attention_n_heads = attention_n_heads
        self.attention_dropout = attention_dropout
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.pos_weight = pos_weight
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.fixed_epochs = fixed_epochs
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network_: FTTransformer | None = None
        self.cat_cardinalities_: list[int] = []
        self.n_iter_ = 0
        self.best_epoch_ = 0
        self.loss_ = float("nan")
        self.history_: list[dict[str, float]] = []

    # ------------------------------------------------------------------ data
    @staticmethod
    def _dense(values: Any) -> NDArray[np.float32]:
        if hasattr(values, "toarray"):
            values = values.toarray()
        return np.asarray(values, dtype=np.float32, order="C")

    def _split_columns(self, X: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Split the matrix into continuous floats and shifted category indices."""

        values = self._dense(X)
        n_cont = values.shape[1] - self.n_cat_features
        continuous = torch.from_numpy(np.ascontiguousarray(values[:, :n_cont]))
        if self.n_cat_features == 0:
            return continuous, None
        codes = values[:, n_cont:].astype(np.int64) + 1  # -1 (unseen) -> 0
        if self.cat_cardinalities_:
            upper = np.asarray(self.cat_cardinalities_, dtype=np.int64) - 1
            codes = np.minimum(codes, upper)
        return continuous, torch.from_numpy(np.ascontiguousarray(codes))

    # ----------------------------------------------------------------- model
    def _build_network(self, n_cont: int) -> FTTransformer:
        kwargs = FTTransformer.get_default_kwargs(n_blocks=self.n_blocks)
        kwargs.update(
            d_block=self.d_block,
            attention_n_heads=self.attention_n_heads,
            attention_dropout=self.attention_dropout,
            ffn_dropout=self.ffn_dropout,
            residual_dropout=self.residual_dropout,
        )
        kwargs.pop("_is_default", None)
        return FTTransformer(
            n_cont_features=n_cont,
            cat_cardinalities=self.cat_cardinalities_,
            d_out=1,
            **kwargs,
        ).to(self.device)

    def _logits(
        self,
        network: nn.Module,
        continuous: torch.Tensor,
        categorical: torch.Tensor | None,
        batch_size: int,
    ) -> NDArray[np.float32]:
        use_amp = self.device.type == "cuda"
        network.eval()
        outputs: list[NDArray[np.float32]] = []
        with torch.no_grad():
            for start in range(0, len(continuous), batch_size):
                stop = start + batch_size
                x_cont = continuous[start:stop].to(self.device)
                x_cat = None if categorical is None else categorical[start:stop].to(self.device)
                with torch.autocast(
                    device_type=self.device.type, dtype=torch.float16, enabled=use_amp
                ):
                    logits = network(x_cont, x_cat)
                outputs.append(logits.float().cpu().numpy().reshape(-1))
        return np.concatenate(outputs)

    # ------------------------------------------------------------------- fit
    def fit(self, X: Any, y: Any, X_val: Any = None, y_val: Any = None) -> Self:
        """Train with weighted BCE.

        Stopping rule, in order of preference:
        1. ``X_val``/``y_val`` given: early stop on validation PR-AUC.
        2. ``fixed_epochs`` set: train exactly that many epochs (final refit).
        3. Otherwise hold out a random ``val_fraction`` of the rows for early
           stopping. This fallback is not grouped by crossing, so prefer 1 or 2.
        """

        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        targets = np.asarray(y, dtype=np.float32).reshape(-1)
        values = self._dense(X)
        if X_val is None and self.fixed_epochs is None:
            idx_train, idx_val = train_test_split(
                np.arange(len(targets)),
                test_size=self.val_fraction,
                stratify=targets,
                random_state=self.random_state,
            )
            X_val, y_val = values[idx_val], targets[idx_val]
            values, targets = values[idx_train], targets[idx_train]

        n_cont = values.shape[1] - self.n_cat_features
        if self.n_cat_features:
            train_codes = values[:, n_cont:].astype(np.int64) + 1
            self.cat_cardinalities_ = [int(c) + 1 for c in train_codes.max(axis=0)]
        continuous, categorical = self._split_columns(values)
        labels = torch.from_numpy(targets.reshape(-1, 1))

        use_validation = X_val is not None
        if use_validation:
            val_continuous, val_categorical = self._split_columns(X_val)
            val_labels = np.asarray(y_val, dtype=np.float32).reshape(-1)

        network = self._build_network(n_cont)
        optimizer = torch.optim.AdamW(
            network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        weight = None if self.pos_weight is None else torch.tensor(
            [self.pos_weight], dtype=torch.float32, device=self.device
        )
        loss_function = nn.BCEWithLogitsLoss(pos_weight=weight)
        use_amp = self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        generator = torch.Generator().manual_seed(self.random_state)

        n_epochs = self.max_epochs if use_validation or self.fixed_epochs is None else self.fixed_epochs
        best_score, best_state, stale = -np.inf, None, 0
        self.history_ = []

        for epoch in range(n_epochs):
            network.train()
            epoch_loss = 0.0
            order = torch.randperm(len(continuous), generator=generator)
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                x_cont = continuous[batch].to(self.device)
                x_cat = None if categorical is None else categorical[batch].to(self.device)
                batch_labels = labels[batch].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=self.device.type, dtype=torch.float16, enabled=use_amp
                ):
                    logits = network(x_cont, x_cat)
                    loss = loss_function(logits.float(), batch_labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.detach().item() * len(batch)
            epoch_loss /= len(order)
            self.n_iter_ = epoch + 1
            self.loss_ = epoch_loss
            record = {"epoch": epoch + 1, "train_loss": epoch_loss}

            if use_validation:
                scores = self._logits(network, val_continuous, val_categorical, self.batch_size * 4)
                score = (
                    float(average_precision_score(val_labels, scores))
                    if val_labels.sum() > 0
                    else 0.0
                )
                record["val_pr_auc"] = score
                if score > best_score:
                    best_score, stale = score, 0
                    self.best_epoch_ = epoch + 1
                    best_state = copy.deepcopy(network.state_dict())
                else:
                    stale += 1
                    if stale >= self.patience:
                        self.history_.append(record)
                        break
            self.history_.append(record)

        if best_state is not None:
            network.load_state_dict(best_state)
        else:
            self.best_epoch_ = self.n_iter_
        self.network_ = network
        print(
            f"FT-Transformer | device={self.device} | epochs={self.n_iter_} | "
            f"best_epoch={self.best_epoch_} | train_loss={self.loss_:.5f}"
            + (f" | val_pr_auc={best_score:.4f}" if use_validation else "")
        )
        return self

    def fit_with_validation(self, X: Any, y: Any, X_val: Any, y_val: Any) -> Self:
        """Signature used by ``tune_model``'s ``fit_model`` callback."""

        return self.fit(X, y, X_val, y_val)

    # --------------------------------------------------------------- predict
    def predict_proba(self, X: Any) -> NDArray[np.float32]:
        """Return two-column probabilities for the negative and positive classes."""

        if self.network_ is None:
            raise RuntimeError("Call fit before predict_proba.")
        continuous, categorical = self._split_columns(X)
        logits = self._logits(self.network_, continuous, categorical, self.batch_size * 4)
        positive = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        return np.column_stack([1.0 - positive, positive]).astype(np.float32, copy=False)
