# Kaggle (optional)

Only needed to embed a very large corpus or run a local 7B LLM. Expected budget: 0–5 GPU h.

```bash
kaggle kernels init -p kaggle/        # creates kernel-metadata.json (id: rajucode/<slug>)
kaggle kernels push -p kaggle/ --accelerator NvidiaTeslaT4
kaggle kernels status rajucode/<slug>
kaggle kernels output rajucode/<slug> -p kaggle/out
```
