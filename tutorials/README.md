# Tutorials

Runnable worked examples. Each subdirectory holds a USRegistrator-compatible `config.yaml`
plus a small callable that resolves it against wherever your data lives — so an example is
something you *run*, not a set of instructions that drift out of date.

```bash
python main.py --list-tutorials
python main.py --tutorial <name> --data-root /path/to/data
```

```python
from tutorials import run_tutorial
run_tutorial("<name>", data_root="/path/to/data", epochs=5)
```

| Tutorial | Topic |
|---|---|
| [`trusted_ct_us_localnet`](trusted_ct_us_localnet/) | Multimodal CT→US kidney registration with LocalNet3D, and the cross-modal preprocessing trap that silently halves your starting Dice. |

Every tutorial callable accepts `check_only=True` (`--check-only`) to run its data sanity
check and stop before training. Use it — it costs seconds and answers "is this my model or
my data?".

See [`trusted_ct_us_localnet/README.md`](trusted_ct_us_localnet/README.md#adding-your-own-tutorial)
for how to add one.
