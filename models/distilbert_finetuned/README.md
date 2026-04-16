DistilBERT artifacts are intentionally minimized in this repository for GitHub size limits.

Excluded by default:
- `checkpoint-*` training snapshots
- `*.safetensors`
- `*.pt`, `*.pth`, `*.bin`

To use DistilBERT inference in this project, provide a local model directory at:
- `models/distilbert_finetuned`

For the public GitHub repo, keep the weight file out of source control. For local demos or other private runtimes, copy the weights onto the machine after cloning the repo.

At minimum, the directory should contain model weights and tokenizer/config files compatible with:
- `AutoTokenizer.from_pretrained(...)`
- `AutoModelForSequenceClassification.from_pretrained(...)`

If weights are missing, the model server automatically falls back to the TF-IDF baseline model.
