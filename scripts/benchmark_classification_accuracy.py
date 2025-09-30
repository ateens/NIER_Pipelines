#!/usr/bin/env python3
"""
TS2Vec vs T-Rep 레이블 추론 능력 비교 (이상 탐지 성능)

목표: class=3 (베이스라인 이상) 데이터를 얼마나 잘 찾아내는가?

방법:
1. TEST 데이터의 class 레이블을 숨김
2. ChromaDB에서 가장 가까운 k개의 이웃 검색
3. 이웃들의 class로 예측 (majority voting)
4. 실제 class와 비교하여 정확도 측정
"""

import sys
import os
sys.path.insert(0, '/home/0_code/NIER_Pipelines')

import chromadb
from chromadb.config import Settings
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import logging
from datetime import datetime
from collections import Counter
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from NIERModules.chroma_ts2vec import Ts2VecEmbedding
from NIERModules.chroma_trep import TRepEmbedding

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 전역 설정
DB_PATH = "/home/0_code/RAG/ChromaDB/ts2vec_embedding_function/NIER_DB"
TEST_DATA_PATH = "/home/0_code/RAG/ts2vec/datasets/NIER/NIER_v2_TEST.csv"
ELEMENTS = ["SO2", "O3", "CO", "NO", "NO2"]
DEVICE = "cuda"

# 모델 경로
TS2VEC_MODEL_PATHS = {
    "SO2": "/home/0_code/RAG/ts2vec/training/v0_1_1__v0_1_1_SO2_20250427_213952/model.pkl",
    "O3": "/home/0_code/RAG/ts2vec/training/v0_1_1__v0_1_1_O3_20250427_213507/model.pkl",
    "CO": "/home/0_code/RAG/ts2vec/training/v0_1_1__v0_1_1_CO_20250427_210220/model.pkl",
    "NO": "/home/0_code/RAG/ts2vec/training/v0_1_1__v0_1_1_NO_20250427_212025/model.pkl",
    "NO2": "/home/0_code/RAG/ts2vec/training/v0_1_1__v0_1_1_NO2_20250427_212740/model.pkl"
}

TREP_MODEL_PATHS = {
    "SO2": "/home/0_code/NIER_Pipelines/NIERModules/chroma_trep/model_pkl/model_SO2.pt",
    "O3": "/home/0_code/NIER_Pipelines/NIERModules/chroma_trep/model_pkl/model_O3.pt",
    "CO": "/home/0_code/NIER_Pipelines/NIERModules/chroma_trep/model_pkl/model_CO.pt",
    "NO": "/home/0_code/NIER_Pipelines/NIERModules/chroma_trep/model_pkl/model_NO.pt",
    "NO2": "/home/0_code/NIER_Pipelines/NIERModules/chroma_trep/model_pkl/model_NO2.pt"
}


class ClassificationBenchmark:
    def __init__(self, k_neighbors=5):
        self.k_neighbors = k_neighbors
        
        self.client = chromadb.PersistentClient(
            path=DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.ts2vec_collection = self.client.get_collection("time_series_collection")
        self.trep_collection = self.client.get_collection("time_series_collection_trep")
        
        # 테스트 데이터 로드
        logger.info(f"테스트 데이터 로드: {TEST_DATA_PATH}")
        self.test_data = pd.read_csv(TEST_DATA_PATH)
        
        # class 열이 있는지 확인
        if 'class' not in self.test_data.columns:
            raise ValueError("TEST 데이터에 'class' 컬럼이 없습니다!")
        
        logger.info(f"  전체 레코드: {len(self.test_data)}")
        logger.info(f"  클래스 분포:")
        class_counts = self.test_data['class'].value_counts()
        for cls, count in sorted(class_counts.items()):
            logger.info(f"    class={cls}: {count}개")
        
        # 모델 초기화
        self.ts2vec_models = self._init_ts2vec_models()
        self.trep_models = self._init_trep_models()
        
        # 결과 저장
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "test_data": TEST_DATA_PATH,
                "k_neighbors": k_neighbors,
                "device": DEVICE
            },
            "ts2vec": {},
            "trep": {},
            "comparison": {}
        }
    
    def _init_ts2vec_models(self):
        """TS2Vec 모델 초기화"""
        logger.info("TS2Vec 모델 초기화...")
        models = {}
        for elem in ELEMENTS:
            models[elem] = Ts2VecEmbedding(
                weight_path=TS2VEC_MODEL_PATHS[elem],
                device=DEVICE,
                encoding_window='full_series'
            )
        logger.info("TS2Vec 모델 초기화 완료")
        return models
    
    def _init_trep_models(self):
        """T-Rep 모델 초기화"""
        logger.info("T-Rep 모델 초기화...")
        models = {}
        for elem in ELEMENTS:
            models[elem] = TRepEmbedding(
                weight_path=TREP_MODEL_PATHS[elem],
                device=DEVICE,
                encoding_window='full_series',
                time_embedding='t2v_sin'
            )
        logger.info("T-Rep 모델 초기화 완료")
        return models
    
    def predict_class_knn(self, query_embedding, collection, element, k):
        """
        K-NN을 사용하여 클래스 예측
        
        Args:
            query_embedding: 쿼리 임베딩
            collection: ChromaDB collection
            element: 원소 (SO2, O3, ...)
            k: 이웃 수
        
        Returns:
            predicted_class: 예측된 클래스
            neighbor_classes: 이웃들의 클래스 리스트
        """
        try:
            # 가장 가까운 k개 검색
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                where={"element": element},
                include=['metadatas']
            )
            
            if not results['metadatas'] or not results['metadatas'][0]:
                return None, []
            
            # 이웃들의 클래스 추출
            neighbor_classes = []
            for metadata in results['metadatas'][0]:
                if 'class' in metadata:
                    try:
                        cls = int(metadata['class'])
                        neighbor_classes.append(cls)
                    except (ValueError, TypeError):
                        continue
            
            if not neighbor_classes:
                return None, []
            
            # Majority voting
            class_counts = Counter(neighbor_classes)
            predicted_class = class_counts.most_common(1)[0][0]
            
            return predicted_class, neighbor_classes
            
        except Exception as e:
            logger.error(f"예측 에러: {e}")
            return None, []
    
    def benchmark_classification(self, model_name, models, collection, max_samples_per_element=100):
        """
        분류 성능 벤치마크
        
        Args:
            model_name: "TS2Vec" 또는 "T-Rep"
            models: 모델 딕셔너리
            collection: ChromaDB collection
            max_samples_per_element: 원소당 최대 샘플 수
        """
        logger.info("=" * 60)
        logger.info(f"{model_name} 레이블 추론 능력 벤치마크 (k={self.k_neighbors})")
        logger.info("=" * 60)
        
        all_predictions = []
        all_true_labels = []
        element_results = {}
        
        for elem in ELEMENTS:
            logger.info(f"\n원소: {elem}")
            
            # 해당 원소의 테스트 데이터 추출
            elem_data = self.test_data[self.test_data['element'] == elem].copy()
            
            if len(elem_data) == 0:
                logger.warning(f"  {elem}: 테스트 데이터 없음")
                continue
            
            # 샘플 수 제한
            if len(elem_data) > max_samples_per_element:
                elem_data = elem_data.sample(n=max_samples_per_element, random_state=42)
            
            logger.info(f"  테스트 샘플 수: {len(elem_data)}")
            
            predictions = []
            true_labels = []
            
            for idx, row in tqdm(elem_data.iterrows(), total=len(elem_data), 
                                desc=f"{model_name} {elem}", leave=False):
                try:
                    # 임베딩 생성
                    embedding = models[elem]([row['values']])[0]
                    
                    # 클래스 예측
                    pred_class, neighbor_classes = self.predict_class_knn(
                        embedding, collection, elem, self.k_neighbors
                    )
                    
                    if pred_class is not None:
                        predictions.append(pred_class)
                        true_labels.append(int(row['class']))
                    
                except Exception as e:
                    logger.error(f"샘플 처리 에러 ({idx}): {e}")
                    continue
            
            if len(predictions) > 0:
                # 원소별 성능 계산
                accuracy = accuracy_score(true_labels, predictions)
                
                # class=3 (이상)에 대한 Recall (가장 중요!)
                recall_class3 = recall_score(true_labels, predictions, labels=[3], average='macro', zero_division=0)
                
                # 전체 precision, recall, f1
                precision = precision_score(true_labels, predictions, average='macro', zero_division=0)
                recall = recall_score(true_labels, predictions, average='macro', zero_division=0)
                f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)
                
                # Confusion matrix
                cm = confusion_matrix(true_labels, predictions)
                
                element_results[elem] = {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                    "recall_class3": float(recall_class3),
                    "confusion_matrix": cm.tolist(),
                    "n_samples": len(predictions)
                }
                
                logger.info(f"  정확도: {accuracy:.4f}")
                logger.info(f"  Precision: {precision:.4f}")
                logger.info(f"  Recall: {recall:.4f}")
                logger.info(f"  F1-Score: {f1:.4f}")
                logger.info(f"  ⭐ class=3 Recall: {recall_class3:.4f} (이상 탐지 성능)")
                
                all_predictions.extend(predictions)
                all_true_labels.extend(true_labels)
        
        # 전체 성능 계산
        if len(all_predictions) > 0:
            overall_accuracy = accuracy_score(all_true_labels, all_predictions)
            overall_precision = precision_score(all_true_labels, all_predictions, average='macro', zero_division=0)
            overall_recall = recall_score(all_true_labels, all_predictions, average='macro', zero_division=0)
            overall_f1 = f1_score(all_true_labels, all_predictions, average='macro', zero_division=0)
            overall_recall_class3 = recall_score(all_true_labels, all_predictions, labels=[3], average='macro', zero_division=0)
            
            # Classification report
            report = classification_report(all_true_labels, all_predictions, zero_division=0, output_dict=True)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"{model_name} 전체 성능")
            logger.info(f"{'='*60}")
            logger.info(f"전체 정확도: {overall_accuracy:.4f}")
            logger.info(f"전체 Precision: {overall_precision:.4f}")
            logger.info(f"전체 Recall: {overall_recall:.4f}")
            logger.info(f"전체 F1-Score: {overall_f1:.4f}")
            logger.info(f"⭐ class=3 Recall: {overall_recall_class3:.4f} (이상 탐지 성능)")
            
            overall_results = {
                "accuracy": float(overall_accuracy),
                "precision": float(overall_precision),
                "recall": float(overall_recall),
                "f1_score": float(overall_f1),
                "recall_class3": float(overall_recall_class3),
                "classification_report": report,
                "n_samples": len(all_predictions),
                "by_element": element_results
            }
            
            return overall_results
        
        return None
    
    def run_benchmark(self, max_samples_per_element=100):
        """벤치마크 실행"""
        logger.info("🚀 레이블 추론 능력 벤치마크 시작")
        logger.info(f"날짜: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"k-neighbors: {self.k_neighbors}")
        logger.info(f"원소: {', '.join(ELEMENTS)}")
        logger.info("")
        
        try:
            # TS2Vec 벤치마크
            ts2vec_results = self.benchmark_classification(
                "TS2Vec", 
                self.ts2vec_models, 
                self.ts2vec_collection,
                max_samples_per_element
            )
            if ts2vec_results:
                self.results["ts2vec"] = ts2vec_results
            
            # T-Rep 벤치마크
            trep_results = self.benchmark_classification(
                "T-Rep", 
                self.trep_models, 
                self.trep_collection,
                max_samples_per_element
            )
            if trep_results:
                self.results["trep"] = trep_results
            
            # 비교
            if ts2vec_results and trep_results:
                self.results["comparison"] = {
                    "accuracy_improvement": float(trep_results["accuracy"] - ts2vec_results["accuracy"]),
                    "f1_improvement": float(trep_results["f1_score"] - ts2vec_results["f1_score"]),
                    "recall_class3_improvement": float(trep_results["recall_class3"] - ts2vec_results["recall_class3"]),
                }
                
                logger.info("\n" + "=" * 60)
                logger.info("📊 비교 결과")
                logger.info("=" * 60)
                logger.info(f"정확도 향상: {self.results['comparison']['accuracy_improvement']:+.4f}")
                logger.info(f"F1-Score 향상: {self.results['comparison']['f1_improvement']:+.4f}")
                logger.info(f"⭐ class=3 Recall 향상: {self.results['comparison']['recall_class3_improvement']:+.4f}")
            
        except Exception as e:
            logger.error(f"벤치마크 실행 중 에러: {e}")
            import traceback
            traceback.print_exc()
        
        return self.results
    
    def save_results(self, output_path="/home/0_code/NIER_Pipelines/scripts/classification_benchmark_results.json"):
        """결과 저장"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        logger.info(f"\n✅ 결과 저장: {output_path}")
    
    def generate_report(self, output_path="/home/0_code/NIER_Pipelines/scripts/classification_benchmark_report.md"):
        """마크다운 리포트 생성"""
        ts2vec = self.results.get("ts2vec", {})
        trep = self.results.get("trep", {})
        comparison = self.results.get("comparison", {})
        
        report = f"""# TS2Vec vs T-Rep 레이블 추론 능력 비교 (이상 탐지 성능)

**생성 일시**: {self.results['timestamp']}  
**k-neighbors**: {self.k_neighbors}  
**테스트 데이터**: {TEST_DATA_PATH}  
**테스트 원소**: {', '.join(ELEMENTS)}

---

## 1. 전체 성능 비교

| 지표 | TS2Vec | T-Rep | 향상 |
|------|--------|-------|------|
"""
        
        if ts2vec and trep:
            report += f"| 정확도 (Accuracy) | {ts2vec['accuracy']:.4f} | {trep['accuracy']:.4f} | {comparison['accuracy_improvement']:+.4f} |\n"
            report += f"| Precision | {ts2vec['precision']:.4f} | {trep['precision']:.4f} | - |\n"
            report += f"| Recall | {ts2vec['recall']:.4f} | {trep['recall']:.4f} | - |\n"
            report += f"| F1-Score | {ts2vec['f1_score']:.4f} | {trep['f1_score']:.4f} | {comparison['f1_improvement']:+.4f} |\n"
            report += f"| **⭐ class=3 Recall** | **{ts2vec['recall_class3']:.4f}** | **{trep['recall_class3']:.4f}** | **{comparison['recall_class3_improvement']:+.4f}** |\n"
        
        report += "\n**class=3 Recall**: 베이스라인 이상(이상 데이터)를 얼마나 잘 찾아내는가? (높을수록 좋음)\n"
        
        report += "\n---\n\n## 2. 원소별 성능 비교\n\n"
        
        if ts2vec and trep and "by_element" in ts2vec and "by_element" in trep:
            report += "### 2.1 정확도 (Accuracy)\n\n"
            report += "| 원소 | TS2Vec | T-Rep | 향상 |\n|------|--------|-------|------|\n"
            
            for elem in ELEMENTS:
                if elem in ts2vec["by_element"] and elem in trep["by_element"]:
                    ts2vec_acc = ts2vec["by_element"][elem]["accuracy"]
                    trep_acc = trep["by_element"][elem]["accuracy"]
                    improvement = trep_acc - ts2vec_acc
                    report += f"| {elem} | {ts2vec_acc:.4f} | {trep_acc:.4f} | {improvement:+.4f} |\n"
            
            report += "\n### 2.2 class=3 Recall (이상 탐지)\n\n"
            report += "| 원소 | TS2Vec | T-Rep | 향상 |\n|------|--------|-------|------|\n"
            
            for elem in ELEMENTS:
                if elem in ts2vec["by_element"] and elem in trep["by_element"]:
                    ts2vec_recall = ts2vec["by_element"][elem]["recall_class3"]
                    trep_recall = trep["by_element"][elem]["recall_class3"]
                    improvement = trep_recall - ts2vec_recall
                    report += f"| {elem} | {ts2vec_recall:.4f} | {trep_recall:.4f} | {improvement:+.4f} |\n"
            
            report += "\n### 2.3 F1-Score\n\n"
            report += "| 원소 | TS2Vec | T-Rep | 향상 |\n|------|--------|-------|------|\n"
            
            for elem in ELEMENTS:
                if elem in ts2vec["by_element"] and elem in trep["by_element"]:
                    ts2vec_f1 = ts2vec["by_element"][elem]["f1_score"]
                    trep_f1 = trep["by_element"][elem]["f1_score"]
                    improvement = trep_f1 - ts2vec_f1
                    report += f"| {elem} | {ts2vec_f1:.4f} | {trep_f1:.4f} | {improvement:+.4f} |\n"
        
        report += "\n---\n\n## 3. 결론\n\n"
        
        if comparison:
            if comparison["recall_class3_improvement"] > 0:
                report += f"### ✅ T-Rep이 이상 탐지에서 우수\n\n"
                report += f"- class=3 Recall이 **{comparison['recall_class3_improvement']:+.4f}** 향상\n"
                report += f"- 베이스라인 이상 데이터를 더 잘 찾아냄\n"
            else:
                report += f"### ⚠️ TS2Vec이 이상 탐지에서 우수\n\n"
                report += f"- class=3 Recall이 **{comparison['recall_class3_improvement']:+.4f}** 변화\n"
            
            if comparison["accuracy_improvement"] > 0:
                report += f"- 전체 정확도도 **{comparison['accuracy_improvement']:+.4f}** 향상\n"
            
            if comparison["f1_improvement"] > 0:
                report += f"- F1-Score도 **{comparison['f1_improvement']:+.4f}** 향상\n"
        
        report += "\n### 주요 발견\n\n"
        report += "이 벤치마크는 **검색 기반 분류(k-NN)** 방식으로 레이블을 추론합니다.\n"
        report += "- 높은 class=3 Recall = 이상 데이터를 잘 탐지\n"
        report += "- 높은 정확도 = 전체적으로 레이블을 잘 예측\n"
        report += "- F1-Score = Precision과 Recall의 조화평균\n"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"✅ 리포트 저장: {output_path}")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("TS2Vec vs T-Rep 레이블 추론 능력 벤치마크")
    logger.info("=" * 60)
    
    try:
        # k=5로 벤치마크 실행
        benchmark = ClassificationBenchmark(k_neighbors=5)
        results = benchmark.run_benchmark(max_samples_per_element=100)
        
        # 결과 저장
        benchmark.save_results()
        benchmark.generate_report()
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ 벤치마크 완료!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ 벤치마크 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
