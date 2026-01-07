"""
Sequential CRM Pipeline - Model Loader

Creator: Qwen3-0.6B (GGUF)
Styler: Qwen3-4B (GGUF) - 더 복잡한 스타일링용
Polisher: HyperCLOVAX 0.5B (GGUF)
"""

import os
import time
import warnings

# ggml 워닝 억제
os.environ["GGML_METAL_LOG_LEVEL"] = "0"
warnings.filterwarnings("ignore", message=".*not supported.*")
warnings.filterwarnings("ignore", message=".*skipping kernel.*")

# HyperCLOVAX 경로 (3B 우선, 없으면 0.5B)
HYPERCLOVAX_PATH_3B = os.path.expanduser(
    "~/.cache/huggingface/hub/models--cherryDavid--HyperCLOVA-X-SEED-Vision-Instruct-3B-Llamafied-Q4_K_S-GGUF/snapshots/baadee14ff4bbce6cd5613ef3648deb4181e5ab0/hyperclova-x-seed-vision-instruct-3b-llamafied-q4_k_s-imat.gguf"
)
HYPERCLOVAX_PATH_0_5B = os.path.expanduser(
    "~/.cache/huggingface/hub/HyperCLOVAX-GGUF/hyperclovax-seed-text-instruct-0.5b-q4_k_m.gguf"
)

# 3B가 있으면 우선 사용
HYPERCLOVAX_PATH = HYPERCLOVAX_PATH_3B if os.path.exists(HYPERCLOVAX_PATH_3B) else HYPERCLOVAX_PATH_0_5B
QWEN_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf"
)
QWEN_4B_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-4B-GGUF/snapshots/bc640142c66e1fdd12af0bd68f40445458f3869b/Qwen3-4B-Q4_K_M.gguf"
)


class ModelSingleton:
    """모델 싱글톤 부모 클래스"""
    _instance = None
    _model = None
    _path = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _load_model(self, name):
        if self._model is not None:
            return
        
        print(f"[{name}] 모델 로딩 중: {os.path.basename(self._path)}...")
        import llama_cpp
        self._model = llama_cpp.Llama(
            model_path=self._path,
            n_gpu_layers=-1,
            n_ctx=2048,
            n_batch=512,
            verbose=False,
            use_mlock=True,
        )
        print(f"[{name}] 모델 로딩 완료 ✓")
    
    def generate(self, messages, max_tokens=300, temperature=0.5):
        t_start = time.time()
        response = self._model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            repeat_penalty=1.1,
        )
        text = response['choices'][0]['message']['content']
        duration = time.time() - t_start
        return text.strip(), duration


class CreatorModel(ModelSingleton):
    """Qwen3 0.6B (생성용)"""
    _instance = None
    _path = QWEN_PATH
    
    def __init__(self):
        self._load_model("Creator/Qwen-0.6B")


class StylerModel(ModelSingleton):
    """Qwen3 4B (브랜드 스타일링용)"""
    _instance = None
    _path = QWEN_4B_PATH
    
    def __init__(self):
        self._load_model("Styler/Qwen-4B")


class ValidatorModel(ModelSingleton):
    """Qwen3 0.6B (판별용)"""
    _instance = None
    _path = QWEN_PATH
    
    def __init__(self):
        self._load_model("Validator/Qwen")


class PolisherModel(ModelSingleton):
    """HyperCLOVAX 0.5B (최종 윤문용 - 한국어 특화)"""
    _instance = None
    _path = HYPERCLOVAX_PATH
    
    def __init__(self):
        self._load_model("Polisher/HCX")


def get_creator():
    return CreatorModel()

def get_styler():
    return StylerModel()

def get_validator():
    return ValidatorModel()

def get_polisher():
    return PolisherModel()
