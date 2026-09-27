# Pathflow Studio 0.9.1

This update fixes dataset/architecture discovery and adds explicit ResNet presets. See `docs/SELECTION_FIX_11.md`. Select BUS-UCLM or ResNet-50 from **Design experiment**, or open **Joint healthcare** directly. Datasets remain visible before download.

# Start with Pathflow Studio 0.9.0

From this project directory, run `bash setup_local.sh`, then `PORT=5001 .venv-pathflow/bin/python run.py`. Use that same interpreter in your IDE. Run `.venv-pathflow/bin/python doctor.py` to check imports. For the new architecture choose **Joint healthcare**. Read `docs/JOINT_TASK_10.md` for the experimental conventions, dataset import and explicit geometry limits. Existing MLP/CNN experiments remain on their original pages.

## Earlier installation notes

# Pathflow Studio 0.4.0

## Updating your current installation

Stop the server. Replace the project source files with this release while keeping your existing `workspace/` directory and `.venv` environment. Start `run.py` with the same Python interpreter that already works. Reload the browser.

Open an existing experiment → Geometry → Analyze checkpoint. Open Observatory, select the required scope/pair, then click **Compute missing measurements** to backfill the saved checkpoints. The default BCANw budget of 2,500 nodes covers the default CNN. Larger graphs report budget exhaustion explicitly; raise the budget only when the estimated memory and runtime are acceptable.

## New installation on Ubuntu / WSL

From the extracted `pathflow_studio` directory:

```bash
python3 -m venv .venv
.venv/bin/python bootstrap.py
.venv/bin/python run.py
```

Visit http://127.0.0.1:5000. Begin with Design experiment → Synthetic bars → Download / prepare dataset → Create experiment.

In VS Code or PyCharm, select this same `.venv/bin/python` interpreter before running `run.py`. Python 3.10–3.12 is supported. Full setup details are in README.md.
