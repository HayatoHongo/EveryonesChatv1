```bash
sudo apt update
sudo apt install -y git git-lfs
git lfs install
pip install -U huggingface_hub
```

pip install torch numpy datasets tiktoken

```bash
python3 - << 'EOF'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ShallowU/FineWeb-Edu-10B-Tokens-NPY",
    repo_type="dataset",
    local_dir="/home/ubuntu/virginia-filesystem",
    local_dir_use_symlinks=False,
)
EOF
```
