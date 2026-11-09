# AEGIS-Net: Two-Stage IoT Intrusion Detection on CICIoT2023

Code, benchmark results and paper for **AEGIS-Net**, a tabular neural network for rare intrusion classes, benchmarked against 14 machine-learning and deep-learning models on CICIoT2023 under one identical protocol.

FPT University, Vietnam, course DAP391m, Group 24.
Tran Le Nam Kha, Duong Khac Huy, Bui Ta Thanh Quang, Nguyen Tuan Anh, Le Nhat Tung.

## Main result (CICIoT2023, 5 seeds, 2,924,221 test rows, macro F1, raw decision)

| Task | AEGIS-Net | Best other model |
| --- | --- | --- |
| Stage 1: benign vs attack | **0.9267** | 0.9256 (CatBoost) |
| Stage 2: attack family | **0.7446** | 0.7423 (XGBoost) |
| 8-class | **0.7260** | 0.7228 (XGBoost) |

AEGIS-Net ranks first on all three tasks, 0.011 to 0.023 ahead of the best other neural network. The margin over the best tree model is small (0.001 to 0.003), and after a shared calibration step XGBoost matches it on the 8-class task. The ablation shows the bin embedding and the shrinkage carry the gain; the ensemble adds nothing measurable on this dataset. Stages 1 and 2 are evaluated as separate tasks, not as an end-to-end cascade. See the paper for details and limitations.

## Repository layout

| Path | Content |
| --- | --- |
| `aegis_net/` | model, 15 baselines, training loop, evaluation |
| `main.py` | entry point: prepares data, trains all models, writes tables |
| `tools/` | data preparation (NSL-KDD, UNSW-NB15), ablation, split check, synthetic data |
| `tests/` | unit tests |
| `notebooks/Aegis_net_colab.ipynb` | Google Colab notebook (smoke test, three datasets, ablation) |
| `results/ciciot2023/` | CICIoT2023 results: summary tables, per-class F1, per-seed runs |
| `results/unsw_nb15/`, `results/nsl_kdd/` | earlier runs, reduced protocol |
| `paper/` | LaTeX source, PDF and figure of the paper |
| `docs/AEGIS_NET_MODEL_README.md` | detailed description of the model and the benchmark protocol |

## Quick start

```bash
pip install -r requirements.txt

# smoke test (about 2 minutes, synthetic data)
python tools/make_synthetic_data.py --out synthetic --rows 20000
python main.py --data_path synthetic --out smoke --epochs 2 --seeds 42 \
    --fit_rows 8000 --models LogReg XGBoost Aegis_net --tasks multiclass
```

## Reproduce the CICIoT2023 benchmark

The dataset is not included (size). Download the CSV files from the CIC IoT Dataset 2023 page of the University of New Brunswick (see the paper, Neto et al., Sensors 23(13):5941, 2023), put them in one folder, then run the official protocol (500,000 fit rows, 5 seeds, 20 epochs; many hours, a GPU is recommended):

```bash
python main.py --data_path /path/to/CICIoT2023 --out outputs/ciciot2023 --cache_dir cache_ciciot
python main.py --report_only --out outputs/ciciot2023     # rebuild tables from finished runs
```

The 34 raw labels are grouped into 8 families automatically. Finished runs are skipped when the command is repeated, so an interrupted run resumes. Add `--models XGBoost LightGBM CatBoost TabM CNN1D Aegis_net --fit_rows 100000 --seeds 42 43 44` for a lighter run.

## Paper

`paper/AegisNet_CICIoT2023_paper.tex` (compile with `pdflatex`; needs `fig_perclass.pdf` in the same folder). `paper/make_figs.py <output_dir>` regenerates the figure.

## Citation

If you use this work, cite the paper once it is published. [To be completed by the authors.]

## License

[To be chosen by the authors, for example MIT.]
