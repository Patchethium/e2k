"""
Shrinks the S2S model by iterative cut-and-distill, adapted from the
"Shrinking gradually" recipe of Kyutai's Pocket TTS small models:
cut the width a little (PCA rotation where the space is only read linearly,
unit selection behind GRU gates and tanh), distill from the original teacher
to convergence, repeat.

Run from `e2k_train/`:
    uv run shrink.py --schedule 192,128,96,64 --baselines
"""

import os

# time numpy inference single threaded, set before numpy is imported
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import time
import argparse
from os import path

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from e2k.inference import S2S, greedy as np_greedy
from e2k_train.train import Model, Words, load_data, loader, train_ce, SEED, DIM
from e2k_train.export import model_weights, model_metadata
from e2k_train.eval import score
from e2k_train.hp import kanas, SOS_IDX, EOS_IDX

HEADS = 4


# ---------------------------------------------------------------- data


# ---------------------------------------------------------------- statistics


def activations(model: Model, src, tgt, src_mask, tgt_mask):
    """
    Teacher-forced forward returning every space the cut touches, and the logits.
    Must match `Model.forward`, call it in eval mode.
    """
    D = model.fc.in_features
    W, b = model.attn.in_proj_weight, model.attn.in_proj_bias
    e = model.e_emb(src)
    k = model.k_emb(tgt)[:, :-1]
    h, _ = model.encoder(e)
    c = model.encoder_fc(h)
    p, _ = model.pre_decoder(k)
    q = F.linear(p, W[:D], b[:D])
    key = F.linear(c, W[D : 2 * D], b[D : 2 * D])
    v = F.linear(c, W[2 * D :], b[2 * D :])
    a, _ = model.attn(p, c, c, key_padding_mask=~src_mask)
    h2, _ = model.post_decoder(torch.cat([p, a], dim=-1))
    logits = model.fc(h2)
    acts = {
        # source side
        "E": e,
        "Hf": h[..., :D],
        "Hb": h[..., D:],
        "C": c,
        "Key": key,
        "V": v,
        # decoder side
        "K": k,
        "P": p,
        "Q": q,
        "A": a,
        "H2": h2,
    }
    return acts, logits


SRC_SPACES = {"E", "Hf", "Hb", "C", "Key", "V"}


@torch.no_grad()
def moments(model: Model, batches):
    """
    Uncentered second moments E[x xᵀ] of every space over non-padding positions,
    in float64.
    """
    model.eval()
    sums, counts = {}, {}
    for src, tgt, src_mask, tgt_mask in batches:
        acts, _ = activations(model, src, tgt, src_mask, tgt_mask)
        for name, x in acts.items():
            mask = src_mask if name in SRC_SPACES else tgt_mask[:, :-1]
            x = x[mask].double()
            sums[name] = sums.get(name, 0) + x.T @ x
            counts[name] = counts.get(name, 0) + x.shape[0]
    return {name: sums[name] / counts[name] for name in sums}


# ---------------------------------------------------------------- the cut


def pca(M, d):
    """top-d eigenvectors of a symmetric matrix, [D, d]"""
    _, evecs = torch.linalg.eigh(M)
    return evecs[:, -d:].flip(-1)


def select(score, d):
    """the d highest scoring units as a selection basis, [D, d]"""
    idx = torch.topk(score, d).indices.sort().values
    return torch.eye(len(score), dtype=score.dtype)[:, idx]


def col_energy(*Ws):
    """squared norm of the columns reading each unit, summed over readers"""
    return sum((W**2).sum(0) for W in Ws)


def gates(B):
    """the same hidden basis for the r, z, n gate blocks"""
    return torch.block_diag(B, B, B)


@torch.no_grad()
def shrink(model: Model, d: int, mom, p2k: bool, rotate: bool = True) -> Model:
    """
    Cuts every space to width d. A space with basis B [D, d] has its writers
    left-multiplied by Bᵀ (biases too) and its readers right-multiplied by B.
    """
    D = model.fc.in_features
    assert d <= D and d % HEADS == 0
    dh, dh_new = D // HEADS, d // HEADS
    w = {k: v.double() for k, v in model.state_dict().items()}
    Wq, Wk, Wv = w["attn.in_proj_weight"].split(D)
    bq, bk, bv = w["attn.in_proj_bias"].split(D)
    Wo = w["attn.out_proj.weight"]
    diag = {name: M.diagonal() for name, M in mom.items()}

    def linear_space(name, *readers):
        # only read through linear maps: rotate onto the principal components
        if rotate:
            return pca(mom[name], d)
        return select(diag[name] * col_energy(*readers), d)

    def unit_space(name, *readers):
        # read through gates or tanh: keep whole units
        return select(diag[name] * col_energy(*readers), d)

    B_E = linear_space("E", w["encoder.weight_ih_l0"], w["encoder.weight_ih_l0_reverse"])
    B_K = linear_space("K", w["pre_decoder.weight_ih_l0"])
    B_A = linear_space("A", w["post_decoder.weight_ih_l0"][:, D:])
    B_Hf = unit_space("Hf", w["encoder.weight_hh_l0"], w["encoder_fc.0.weight"][:, :D])
    B_Hb = unit_space(
        "Hb", w["encoder.weight_hh_l0_reverse"], w["encoder_fc.0.weight"][:, D:]
    )
    B_C = unit_space("C", Wk, Wv)
    B_P = unit_space(
        "P", w["pre_decoder.weight_hh_l0"], Wq, w["post_decoder.weight_ih_l0"][:, :D]
    )
    B_H2 = unit_space("H2", w["post_decoder.weight_hh_l0"], w["fc.weight"])

    # per head, so the head boundaries stay aligned
    qk_blocks, v_blocks = [], []
    for h in range(HEADS):
        s = slice(h * dh, (h + 1) * dh)
        qk_blocks.append(select(diag["Q"][s] * diag["Key"][s], dh_new))
        if rotate:
            v_blocks.append(pca(mom["V"][s, s], dh_new))
        else:
            v_blocks.append(select(diag["V"][s] * col_energy(Wo[:, s]), dh_new))
    B_QK = torch.block_diag(*qk_blocks)
    B_V = torch.block_diag(*v_blocks)
    # torch scales the logits by 1/sqrt(head_dim), keep them unchanged
    scale = (dh_new / dh) ** 0.5

    new = {
        "e_emb.weight": w["e_emb.weight"] @ B_E,
        "k_emb.weight": w["k_emb.weight"] @ B_K,
        "encoder_fc.0.weight": B_C.T
        @ w["encoder_fc.0.weight"]
        @ torch.block_diag(B_Hf, B_Hb),
        "encoder_fc.0.bias": B_C.T @ w["encoder_fc.0.bias"],
        "attn.in_proj_weight": torch.cat(
            [scale * B_QK.T @ Wq @ B_P, B_QK.T @ Wk @ B_C, B_V.T @ Wv @ B_C]
        ),
        "attn.in_proj_bias": torch.cat([scale * B_QK.T @ bq, B_QK.T @ bk, B_V.T @ bv]),
        "attn.out_proj.weight": B_A.T @ Wo @ B_V,
        "attn.out_proj.bias": B_A.T @ w["attn.out_proj.bias"],
        "fc.weight": w["fc.weight"] @ B_H2,
        "fc.bias": w["fc.bias"],
    }
    grus = [
        ("encoder", "_l0", B_E, B_Hf),
        ("encoder", "_l0_reverse", B_E, B_Hb),
        ("pre_decoder", "_l0", B_K, B_P),
        ("post_decoder", "_l0", torch.block_diag(B_P, B_A), B_H2),
    ]
    for gru, suffix, B_in, B_h in grus:
        G = gates(B_h)
        new[f"{gru}.weight_ih{suffix}"] = G.T @ w[f"{gru}.weight_ih{suffix}"] @ B_in
        new[f"{gru}.weight_hh{suffix}"] = G.T @ w[f"{gru}.weight_hh{suffix}"] @ B_h
        new[f"{gru}.bias_ih{suffix}"] = G.T @ w[f"{gru}.bias_ih{suffix}"]
        new[f"{gru}.bias_hh{suffix}"] = G.T @ w[f"{gru}.bias_hh{suffix}"]

    student = Model(p2k=p2k, dim=d)
    student.load_state_dict({k: v.float() for k, v in new.items()}, strict=True)
    return student


@torch.no_grad()
def check_equivalence(model: Model, mom, batches, p2k: bool):
    """
    At full width every basis is orthogonal and every selection a permutation,
    so the cut must reproduce the logits. Catches indexing and transpose bugs.
    """
    model.eval()
    D = model.fc.in_features
    for rotate in (True, False):
        cut = shrink(model, D, mom, p2k, rotate=rotate).eval()
        err = 0.0
        for src, tgt, src_mask, tgt_mask in batches:
            a = model(src, tgt, src_mask, tgt_mask)
            b = cut(src, tgt, src_mask, tgt_mask)
            err = max(err, (a - b).abs().max().item())
        print(f"equivalence check (rotate={rotate}): max |Δlogit| = {err:.2e}")
        assert err < 1e-4, "the full-width cut does not reproduce the model"


# ---------------------------------------------------------------- training


def kd_parts(student_logits, teacher_logits, target):
    """KL(teacher || student) and CE(data), per token over non-padding targets"""
    mask = target != 0
    s = F.log_softmax(student_logits[mask], dim=-1)
    t = F.log_softmax(teacher_logits[mask], dim=-1)
    kl = F.kl_div(s, t, log_target=True, reduction="batchmean")
    ce = F.nll_loss(s, target[mask])
    return kl, ce


def kd_loss(student_logits, teacher_logits, target):
    """0.5 KL(teacher || student) + 0.5 CE(data)"""
    kl, ce = kd_parts(student_logits, teacher_logits, target)
    return 0.5 * kl + 0.5 * ce


@torch.no_grad()
def val_loss(model, teacher, val_dl, parts=False):
    """
    token weighted, the KD loss (0.5 KL + 0.5 CE) if a teacher is given, else CE;
    with `parts`, also the KL and CE terms
    """
    model.eval()
    kl_sum, ce_sum, count = 0.0, 0.0, 0
    for src, tgt, src_mask, tgt_mask in val_dl:
        out = model(src, tgt, src_mask, tgt_mask)
        n = (tgt[:, 1:] != 0).sum().item()
        if teacher is not None:
            kl, ce = kd_parts(out, teacher(src, tgt, src_mask, tgt_mask), tgt[:, 1:])
            kl_sum += kl.item() * n
        else:
            ce = F.cross_entropy(out.transpose(1, 2), tgt[:, 1:], ignore_index=0)
        ce_sum += ce.item() * n
        count += n
    kl, ce = kl_sum / count, ce_sum / count
    loss = 0.5 * kl + 0.5 * ce if teacher is not None else ce
    return (loss, kl, ce) if parts else loss


def distill(student, teacher, train_dl, val_dl, lr=5e-4, max_epochs=40, patience=3):
    """trains to convergence on the KD loss, restores the best epoch"""
    teacher.eval()
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=1)
    best, kl, ce = val_loss(student, teacher, val_dl, parts=True)
    bad, history = 0, [best]
    print(f"  epoch 0: val loss {best:.4f} (KL {kl:.4f}, CE {ce:.4f})", flush=True)
    best_state = {k: v.clone() for k, v in student.state_dict().items()}
    for epoch in range(1, max_epochs + 1):
        student.train()
        t0 = time.time()
        for src, tgt, src_mask, tgt_mask in train_dl:
            with torch.no_grad():
                t_out = teacher(src, tgt, src_mask, tgt_mask)
            optimizer.zero_grad()
            out = student(src, tgt, src_mask, tgt_mask)
            loss = kd_loss(out, t_out, tgt[:, 1:])
            loss.backward()
            optimizer.step()
        loss, kl, ce = val_loss(student, teacher, val_dl, parts=True)
        scheduler.step(loss)
        history.append(loss)
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"  epoch {epoch}: val loss {loss:.4f} (KL {kl:.4f}, CE {ce:.4f}) "
            f"lr {lr_now:.1e} ({time.time() - t0:.0f}s)",
            flush=True,
        )
        if loss < best - 1e-4:
            best, bad = loss, 0
            best_state = {k: v.clone() for k, v in student.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    student.load_state_dict(best_state)
    return history


# ---------------------------------------------------------------- evaluation


def decode(ids):
    out = []
    for i in ids:
        if i == EOS_IDX:
            break
        out.append(kanas[i])
    return "".join(out)


@torch.no_grad()
def greedy_batched(model, srcs, max_len=16, batch_size=256):
    """
    Greedy decoding. Batches only words of equal length, so the reverse
    encoder never sees padding, as in the unbatched inference.
    """
    model.eval()
    preds = [None] * len(srcs)
    by_len = {}
    for i, s in enumerate(srcs):
        by_len.setdefault(len(s), []).append(i)
    for idxs in by_len.values():
        for k in range(0, len(idxs), batch_size):
            chunk = idxs[k : k + batch_size]
            src = torch.stack([srcs[i] for i in chunk])
            enc, _ = model.encoder(model.e_emb(src))
            enc = model.encoder_fc(enc)
            tok = torch.full((len(chunk), 1), SOS_IDX)
            h1 = h2 = None
            out = []
            for _ in range(max_len):
                dec, h1 = model.pre_decoder(model.k_emb(tok), h1)
                attn, _ = model.attn(dec, enc, enc)
                x, h2 = model.post_decoder(torch.cat([dec, attn], dim=-1), h2)
                tok = model.fc(x).argmax(-1)
                out.append(tok)
                if (torch.cat(out, dim=1) == EOS_IDX).any(1).all():
                    break
            out = torch.cat(out, dim=1).tolist()
            for i, ids in zip(chunk, out):
                preds[i] = decode(ids)
    return preds


def evaluate(model, base, test_idx, p2k: bool):
    srcs = [base.srcs[i] for i in test_idx]
    refs = [base.data[i]["kata"] for i in test_idx]
    em, cer = score(greedy_batched(model, srcs), refs)
    # the fp16 weights `export.py` writes, through the numpy inference, single threaded
    s2s = S2S(model_weights(model), model_metadata(p2k, accent=False))
    np_srcs = [s.numpy() for s in srcs]
    for s in np_srcs[:20]:
        s2s.forward(s, np_greedy)
    t0 = time.perf_counter()
    np_preds = [decode(s2s.forward(s, np_greedy)[1:]) for s in np_srcs]
    ms = (time.perf_counter() - t0) / len(np_srcs) * 1000
    em_npz, cer_npz = score(np_preds, refs)
    return {
        "em": em,
        "cer": cer,
        "em_npz": em_npz,
        "cer_npz": cer_npz,
        "ms_per_word": ms,
        "n_test": len(test_idx),
    }


# ---------------------------------------------------------------- pipeline


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="vendor/katakana_dict.jsonl")
    parser.add_argument("--p2k", action="store_true")
    parser.add_argument("--schedule", type=str, default="192,128,96,64")
    parser.add_argument("--out", type=str, default="vendor/shrink")
    parser.add_argument(
        "--no-rotate", action="store_true", help="unit selection in every space"
    )
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--baseline-widths", type=str, default="128,64")
    parser.add_argument("--moment-batches", type=int, default=200)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=3, help="early stopping, in epochs")
    # MKL_NUM_THREADS above also caps torch, training wants more
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    p2k = args.p2k
    task = "p2k" if p2k else "c2k"
    os.makedirs(args.out, exist_ok=True)
    results_path = path.join(args.out, "results.json")
    results = json.load(open(results_path)) if path.exists(results_path) else {}

    def save_results():
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    base, train_idx, val_idx, test_idx = load_data(args.data, p2k)
    print(f"split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
    train_dl = loader(Words(base, train_idx, fixed=False), shuffle=True)
    val_dl = loader(Words(base, val_idx, fixed=True), shuffle=False)
    # a fixed sample of training batches for the statistics
    stat_ds = Words(base, train_idx, fixed=True)
    stat_dl = loader(stat_ds, shuffle=True)
    stat_batches = []
    for batch in stat_dl:
        stat_batches.append(batch)
        if len(stat_batches) >= args.moment_batches:
            break

    def stage(name, width, build, record=None):
        """trains (or loads) a model and evaluates it, resumable"""
        pth = path.join(args.out, f"{name}.pth")
        model = Model(p2k=p2k, dim=width)
        if path.exists(pth):
            print(f"== {name}: loading {pth}")
            model.load_state_dict(torch.load(pth))
        else:
            print(f"== {name}: training", flush=True)
            torch.manual_seed(SEED)
            t0 = time.time()
            model, info = build()
            info["train_minutes"] = (time.time() - t0) / 60
            torch.save(model.state_dict(), pth)
            results[name] = {"width": width, "params": n_params(model), **info}
            save_results()
        if "em" not in results.get(name, {}):
            results.setdefault(name, {"width": width, "params": n_params(model)})
            results[name].update(evaluate(model, base, test_idx, p2k))
            save_results()
        r = results[name]
        print(
            f"== {name}: EM {r['em']:.4f} CER {r['cer']:.4f} "
            f"(fp16 EM {r['em_npz']:.4f}) {r['ms_per_word']:.2f} ms/word",
            flush=True,
        )
        return model.eval()

    def build_teacher():
        model = Model(p2k=p2k, dim=DIM)
        return model, {"init": "scratch", "loss": "ce", "history": train_ce(model, train_dl, val_dl)}

    teacher = stage(f"{task}-teacher", DIM, build_teacher)
    for p in teacher.parameters():
        p.requires_grad_(False)

    mom = moments(teacher, stat_batches)
    check_equivalence(teacher, mom, stat_batches[:10], p2k)

    def build_cut(prev, width, rotate):
        def build():
            m = moments(prev, stat_batches)
            student = shrink(prev, width, m, p2k, rotate=rotate)
            after_cut = val_loss(student, teacher, val_dl)
            history = distill(
                student, teacher, train_dl, val_dl,
                max_epochs=args.max_epochs, patience=args.patience,
            )
            info = {
                "init": f"cut from {prev.fc.in_features}",
                "rotate": rotate,
                "loss": "kd",
                "val_after_cut": after_cut,
                "val": min(history),
                "history": history,
            }
            return student, info

        return build

    rotate = not args.no_rotate
    chain = "chain" if rotate else "chain-norot"
    prev = teacher
    schedule = [int(x) for x in args.schedule.split(",")]
    for width in schedule:
        prev = stage(f"{task}-{chain}-{width}", width, build_cut(prev, width, rotate))

    if args.baselines:
        for width in [int(x) for x in args.baseline_widths.split(",")]:

            def build_scratch_ce(width=width):
                model = Model(p2k=p2k, dim=width)
                history = train_ce(model, train_dl, val_dl)
                return model, {"init": "scratch", "loss": "ce", "history": history}

            def build_scratch_kd(width=width):
                model = Model(p2k=p2k, dim=width)
                # random init, so the from-scratch learning rate
                history = distill(model, teacher, train_dl, val_dl, lr=1e-3, max_epochs=args.max_epochs)
                return model, {"init": "scratch", "loss": "kd", "val": min(history), "history": history}

            stage(f"{task}-scratch-ce-{width}", width, build_scratch_ce)
            stage(f"{task}-scratch-kd-{width}", width, build_scratch_kd)
            if width != schedule[0]:
                stage(f"{task}-onecut-{width}", width, build_cut(teacher, width, rotate))

    print(f"\n{'model':<24} {'width':>5} {'params':>9} {'EM':>7} {'CER':>7} {'ms/word':>8}")
    for name, r in results.items():
        if name.startswith(task) and "em" in r:
            print(
                f"{name:<24} {r['width']:>5} {r['params']:>9,} "
                f"{r['em']:>7.4f} {r['cer']:>7.4f} {r['ms_per_word']:>8.2f}"
            )


if __name__ == "__main__":
    main()
