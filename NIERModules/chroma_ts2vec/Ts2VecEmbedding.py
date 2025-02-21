# import torch
import numpy as np
from typing import List, Optional
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
import os
import logging
# import sys
# sys.path.append('.')
from .ts2vec import TS2Vec  # TS2Vec 모델을 사용하기 위해 임포트

logger = logging.getLogger(__name__)

class Ts2VecEmbedding(EmbeddingFunction):
    def __init__(self, weight_path: str, device: str = 'cpu', encoding_window: Optional[str] = 'full_series'):
        """
        :param weight_path: Path to the model file.
        :param device: Device to run the model on.
        """
        self.device = device
        self.model = self._load_model(weight_path)
        self.encoding_window = encoding_window

    def _load_model(self, weight_path: str) -> TS2Vec:
        """
        Loads TS2Vec(Embedder) model from the given weight file.
        :param weight_path: Path to the model file.
        :return: TS2Vec model.
        """
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Model weights not found at {weight_path}")
        
        logger.info(f"Loading model from {weight_path}...")
        model = TS2Vec(input_dims=1)
        model.load(weight_path)
        logger.info("Model loaded successfully.")
        
        return model
        
    def __call__(self, input: Documents) -> Embeddings:
        logger.info("Generating embeddings for the input data...")
        
        # 'values' 열에서 문자열을 시계열 데이터로 변환
        input_data = [self._process_values_string(doc) for doc in input]
        # 시계열 데이터로 변환된 데이터를 numpy 배열로 변환
        input_data = np.array(input_data, dtype=np.float32)
        
        embeddings = self.model.encode(
            data=input_data,
            encoding_window=self.encoding_window,
            batch_size=len(input_data)
        )
        # 평탄화하여 1차원 리스트로 변환
        flattened_embeddings = [embedding.flatten().tolist() for embedding in embeddings]
        
        logger.info("Embeddings generated successfully.")
        return flattened_embeddings
    
    def _process_values_string(self, values_str: str) -> List[List[float]]:
        """
        'values' 열의 문자열을 시계열 데이터로 변환하는 메서드.
        :param values_str: 시계열 데이터가 포함된 문자열
        :return: 시계열 데이터 (리스트의 리스트)
        """
        # 문자열을 콤마로 분리하여 실수 리스트로 변환
        try:
            values = [
                float(val) if float(val) != 999999.0 else float('nan')
                for val in values_str.split(',')
            ]
            return [[val] for val in values]  # TS2Vec은 2D 입력을 기대하므로 리스트의 리스트로 반환
        except ValueError as e:
            logger.error(f"Error parsing values string: {values_str}")
            raise e

    def calculate_similarity(self, embedding1, embedding2):
        """
        두 임베딩 간 유사도를 계산하는 함수. NumPy를 사용하여 유클리디안 거리를 계산.
        :param embedding1: 첫 번째 임베딩 벡터 (NumPy 배열 또는 리스트)
        :param embedding2: 두 번째 임베딩 벡터 (NumPy 배열 또는 리스트)
        :return: 유클리디안 거리로 계산된 유사도
        """
        embedding1 = np.array(embedding1)
        embedding2 = np.array(embedding2)

        # 유클리디안 거리 계산: sqrt(sum((x1 - x2)^2))
        distance = np.linalg.norm(embedding1 - embedding2)
        return distance
