from huggingface_hub import upload_folder, HfApi
import os

repo_id = "HayatoHongo/everyoneschat-checkpoints"
local_dir = "/home/ubuntu/virginia-filesystem/checkpoints"

# repo 作成（既にあればスキップ）
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(
    repo_id=repo_id,
    private=False,   # private にしたいなら True
    exist_ok=True
)

# フォルダ丸ごとアップロード
upload_folder(
    folder_path=local_dir,
    repo_id=repo_id,
    token=os.environ["HF_TOKEN"],
)
