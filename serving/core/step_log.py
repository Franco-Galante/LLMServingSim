"""Per-step batch records, written when ``--step-log`` is given.

One CSV row per completed batch, i.e. per step of an instance: when it ran and
what it computed. It is the input for utilization metrics (MFU, MBU) that the
per-second log cannot give: those need each step's duration together with its
token, attention and KV-read counts.

The file belongs to one simulator process. Rows go to ``<path>.tmp`` and the
file is renamed to ``<path>`` only by ``close()`` on a normal end, so a run that
crashed or was killed never leaves a log that looks complete, and parallel runs
with distinct paths cannot touch each other's files.
"""

import csv
import os

COLUMNS = [
    "instance",          # instance id
    "batch_id",
    "dummy",             # 1 for a DP synchronisation dummy (no requests)
    "start_ns",          # clock when the batch was built and dispatched
    "end_ns",            # clock when the last NPU of the instance finished it
    "npu_end_ns",        # "npu:ns;..." per NPU (pipeline stages, P/D receiver)
    "num_prefill",       # sequences with a prefill chunk (incl. recompute)
    "num_decode",        # decode sequences
    "sequences",         # all sequences in the step (lm_head rows)
    "prefill_tokens",    # tokens in prefill chunks
    "recompute_tokens",  # of all tokens, those computed before (preemption)
    "decode_tokens",     # tokens in decode rows (1 + drafts per sequence)
    "forward_tokens",    # rows the forward runs: the above plus CUDA-graph padding
    "attn_pairs",        # causal query-key pairs: sum q*k + q*(q+1)/2
    "kv_read_tokens",    # KV positions the step's attention reads: sum k + q
    "kv_used_bytes",     # KV bytes allocated on the instance at dispatch
    "gen_tokens",        # output tokens the step produced
]


class StepLog:

    def __init__(self, path):
        self.path = path
        self._tmp = path + ".tmp"
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        self._file = open(self._tmp, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(COLUMNS)

    def record(self, instance_id, batch, finish, gen_tokens):
        npu_end = ";".join(f"{npu}:{t}" for npu, t in sorted(batch.npu_end.items()))
        if not batch.requests:
            # DP dummy: a padded forward with no request behind it.
            row = [instance_id, batch.batch_id, 1, batch.batch_time, finish, npu_end,
                   0, 0, 0, 0, 0, 0, batch.total_len, 0, 0, batch.kv_size, gen_tokens]
        else:
            q_list, k_list = batch.q_list, batch.k_list
            prefill_tokens = sum(batch.prefill_q_list)
            # total_len is the padded forward size by now (CUDA-graph and DP
            # padding raise it), so real tokens come from the per-sequence list.
            row = [instance_id, batch.batch_id, 0, batch.batch_time, finish, npu_end,
                   batch.num_prefill, batch.num_decode, len(q_list),
                   prefill_tokens, batch.recompute_tokens,
                   sum(q_list) - prefill_tokens, batch.total_len,
                   sum(q * k + q * (q + 1) // 2 for q, k in zip(q_list, k_list)),
                   sum(q + k for q, k in zip(q_list, k_list)),
                   batch.kv_size, gen_tokens]
        self._writer.writerow(row)

    def close(self):
        """Finish the file and publish it under its final name."""
        self._file.close()
        os.replace(self._tmp, self.path)
