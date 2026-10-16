import os
import warnings
from typing import List

from backend.domain.interfaces.embedding import BaseEmbedding
from backend.common.config import settings
from backend.common.logger import get_logger

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

class BGEM3Local(BaseEmbedding):
    def __init__(self):
        # 保持离线环境加载模型
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        # 不再硬编码，从 settings 获取设备配置
        cuda_visible_devices = getattr(settings, "EMBEDDING_CUDA_VISIBLE_DEVICES", "")
        if cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

        import torch
        from sentence_transformers import SentenceTransformer

        # 🌟 修正点：增强读取配置的鲁棒性
        configured_device = str(getattr(settings, "EMBEDDING_DEVICE", "auto")).strip().lower()

        # 设备自适应
        if configured_device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif configured_device in {"cuda", "cpu"}:
            self.device = configured_device
        else:
            logger.warning("未知 EMBEDDING_DEVICE=%s，自动回退为 auto", configured_device)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("配置要求使用 cuda，但当前 CUDA 不可用，回退到 cpu")
            self.device = "cpu"

        self.batch_size = int(getattr(settings, "EMBEDDING_BATCH_SIZE", 64))
        model_path = settings.EMBEDDING_MODEL_PATH

        logger.info(
            "Loading offline embedding model: path=%s, device=%s, batch_size=%s",
            model_path,
            self.device,
            self.batch_size,
        )

        try:
            self.model = SentenceTransformer(model_path, device=self.device)

            # 限制在 512 Tokens，保证最高语义浓度
            self.model.max_seq_length = 512
            
            if self.device == "cuda":
                gpu_name = torch.cuda.get_device_name(0)
                logger.info("BGE-M3 loaded on GPU: %s", gpu_name)
            else:
                logger.info("BGE-M3 loaded on CPU")

            logger.info("Embedding max_seq_length=512")
        except Exception as e:
            logger.error("Embedding model load failed: %s", e)
            raise e

    def embed_text(self, text: str) -> List[float]:
        if not text.strip():
            return []
        vector = self.model.encode(
            text, 
            normalize_embeddings=True, 
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return vector.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=self.batch_size,
            convert_to_numpy=True
        )
        return embeddings.tolist()
