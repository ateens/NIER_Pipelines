"""
title: nier_rag_pipeline
author: standard_lee@inha.edu
date: 2025-2-17
version: 1.0.0
description: A pipeline for using TEXT_TO_TS_TO_EMBEDDING(text->ts->embedding) for retrieving neareast neighbors from a database using the ChromaDB.
requirements: chromadb, requests, pandas

!!! Requires Ollma 0.5.0 or higher. !!!
"""
# TODO: 20250217 - I refactored the overall code structure. Rewrite docstrings and comments for each package.

from NIERModules.NIERStation import StationNetwork, GeoSpatialNetwork, StationStats
from NIERModules.chroma_db_handler import get_chromadb_collection_id
from NIERModules.chroma_ts2vec import Ts2VecEmbedding
from NIERModules.ollama_handler import (call_ollama_chat_api,
                                        query_general_question)
from NIERModules.chroma_db_handler import (get_chromadb_collection_id,
                                           query_chromadb_with_filter)
from NIERModules.query_parser import (parse_query,
                                      validate_query)
from NIERModules.postgres_handler import fetch_data
from NIERModules.timeseries_handler import compare_time_series, calculate_z_score
from typing import List, Generator
from pydantic import BaseModel
import os


class Pipeline:
    class Valves(BaseModel):
        VECTOR_DB_HOST: str
        VECTOR_DB_PORT: str
        POSTGRESQL_URL: str
        POSTGRESQL_PORT: str
        POSTGRESQL_DB: str
        POSTGRESQL_USER: str
        POSTGRESQL_PASSWORD: str
        VECTOR_COLLECTION_NAME: str
        OLLAMA_HOST: str
        TEXT_TO_TS_TO_EMBEDDING_MODEL: str
        TASK_MODEL: str
        VECTOR_DB_TOP_K: int
        ADDITIONAL_DAYS: int
        DOUBLE_THE_SEQUENCE: bool

    def __init__(self):
        self.name = "NIER RAG Pipeline"
        self.valves = self.Valves(
            **{
                # Connect to all pipelines
                "pipelines": ["*"],
                "VECTOR_DB_HOST": os.getenv("VECTOR_DB_HOST", "http://localhost"),
                "VECTOR_DB_PORT": os.getenv("VECTOR_DB_PORT", "8000"),
                "VECTOR_COLLECTION_NAME": os.getenv("VECTOR_COLLECTION_NAME", "time_series_collection"),
                "VECTOR_DB_TOP_K": os.getenv("VECTOR_DB_TOP_K", 10),
                "POSTGRESQL_URL": os.getenv("POSTGRESQL_URL", "e2m3.iptime.org"),
                "POSTGRESQL_PORT": os.getenv("POSTGRESQL_PORT", "5432"),
                "POSTGRESQL_DB": os.getenv("POSTGRESQL_DB", "airinfo"),
                "POSTGRESQL_USER": os.getenv("POSTGRESQL_USER", "ais"),
                "POSTGRESQL_PASSWORD": os.getenv("POSTGRESQL_PASSWORD", "aisUser1@@"),
                "OLLAMA_HOST": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
                "TEXT_TO_TS_TO_EMBEDDING_MODEL": os.getenv("TEXT_TO_TS_TO_EMBEDDING_MODEL", "Bllossom-8B:latest"),
                "TASK_MODEL": os.getenv("TASK_MODEL", "Bllossom-8B:latest"),
                "ADDITIONAL_DAYS": os.getenv("ADDITIONAL_DAYS", 14),
                "DOUBLE_THE_SEQUENCE": os.getenv("DOUBLE_THE_SEQUENCE", False)
            }
        )
        
        self.db_path = "/home/1_Dataset/NIER/download/21222324/2024.csv"
        self.embedding_model_path = "/home/0_code/OpenWebUI/pipelines/NIERModules/chroma_ts2vec/model.pkl"
        self.embedding_function = Ts2VecEmbedding(
            weight_path=self.embedding_model_path, device="cpu")
        self.chromadb_url = "http://localhost:8000/api/v1/collections"
        self.collection_id = None
        
        # Init station related objects
        self.station_network = self._initialize_station_network()
        self.geospatial_network = self._initialize_geospatial_network()
        self.station_stats = self._initialize_station_stats()

    def _initialize_station_network(self) -> StationNetwork:
        """
        Initializes the StationNetwork object.
        """
        print("[Info] Initializing StationNetwork...")

        return StationNetwork()

    def _initialize_geospatial_network(self) -> GeoSpatialNetwork:
        """
        Initializes the GeoSpatialNetwork object.
        """
        print("[Info] Initializing GeoSpatialNetwork...")

        return GeoSpatialNetwork()

    def _initialize_station_stats(self) -> StationStats:
        """
        Initializes the StationStats object.
        """
        print("[Info] Initializing StationStats...")

        return StationStats()

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        self.collection_id = get_chromadb_collection_id(
            self.valves.VECTOR_DB_HOST, self.valves.VECTOR_DB_PORT, self.valves.VECTOR_COLLECTION_NAME)

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")

    def embed_values(self, values: str) -> dict:
        print("###EMBED_VALUES###")
        """Step 3: Embed time-series values"""
        print(f"Embedding values...{values}")
        embedding = self.embedding_function([values])[0]

        return embedding.tolist()

    def build_explain_prompt(self, original_data, station_comparison_results, similar_data):
        """
        Build a comprehensive prompt for LLM using original data and comparison results.
        """
        print("###BUILD_EXPLAIN_PROMPT###")
        original_element = original_data['element']

        station_results = []
        confidence_scores = []

        for idx, res in enumerate(station_comparison_results):

            # Get the original station and the compared station
            station_a = original_data['region']
            station_b = res['compared_with']
            # Get the similarity score from the StationNetwork
            avg_distance, standard_deviation = self.station_network.search_similarity(
                station_a, station_b, original_element, '6h')
            print(
                f"distance between {station_a} - {station_b} average_distance {avg_distance}, calculated_distance {res['distance']}")
            z_score = calculate_z_score(
                res['distance'], avg_distance, standard_deviation)

            confidence = max(0, 1 - abs(z_score) / 3)  # Z-score 절대값을 사용하여 정규화
            confidence_scores.append(confidence)

            # Interpret Z-score

            if z_score < 1.0:
                z_score_comment = "와의 유사도가 평균 범위 내에 있습니다."
            elif z_score < 2.0:
                z_score_comment = "와의 유사도가 약간 낮습니다."
            elif z_score < 3.0:
                z_score_comment = "와의 유사도가 평균보다 낮으므로 이상 징후 의심이 됩니다."
            else:
                z_score_comment = "와의 유사도가 평균보다 크게 낮아 베이스라인 이상 징후가 강하게 의심됩니다."

            station_results.append(
                f"- {idx+1}순위 연관 측정소 {res['compared_with']} {z_score_comment}"
            )

        # Average confidence of adjacent stations
        avg_confidence = sum(confidence_scores) / \
            len(confidence_scores) if confidence_scores else 1.0

        station_section = "\n".join(
            station_results) if station_results else "연관 측정소가 없습니다."

        abnormal_weight = 0.0
        normal_weight = 0.0
        total_weight = 0.0
        min_distance = float('inf')
        similar_results = []

        for meta, distance in zip(similar_data.get('metadatas', []), similar_data.get('distances', [])):

            if distance < min_distance:
                min_distance = distance

            state = '정상 데이터' if meta.get('class') == 0 else '베이스라인 이상 데이터'
            weight = 1 / (distance + 1e-6)
            if distance < 2.0:
                similarity = "유사"
            elif distance < 5.0:
                similarity = "다름"
            else:
                similarity = "매우 다름"
            if meta.get('class') == 0:
                normal_weight += weight
            elif meta.get('class') == 3:
                abnormal_weight += weight
            total_weight += weight

            similar_results.append(
                f"- 기존 판정 결과 {meta.get('region', 'N/A')} (성분: {meta.get('element', 'N/A')}, "
                f"상태: {state}, 유사도: {similarity}, "
                f"측정일시: {meta.get('original_start', 'N/A')} ~ {meta.get('original_end', 'N/A')})"
            )
            # f"L2 Distance: {distance:.4f}, 상태: {state}, 유사도: {similarity}, "

        similar_results_text = "\n".join(
            similar_results) if similar_results else "유사 기존 판정 결과가 없습니다."

        extreme_difference_comment = ""

        # If the L2 distance is too high, do not use probability
        if min_distance is not float('inf'):
            if min_distance > 100.0:
                # This means the data is too different from the usual data
                # Almost unseen data
                final_abnormal_probability = 100.0
                final_normal_probability = 0.0
                extreme_difference_comment = (
                    "기존 데이터와 유사성을 찾기 어려운 거의 관측되지 않은 데이터일 가능성이 높습니다."
                    "따라서, 비정상 데이터로 판단할 수 있습니다."
                )
            else:
                # Calculate abnormal and normal probabilities (Weighted)
                abnormal_probability = (
                    abnormal_weight / total_weight * 100) if total_weight > 0 else 0
                normal_probability = (
                    normal_weight / total_weight * 100) if total_weight > 0 else 0

                # 최종 확률 계산: 벡터DB 기반 판정 결과와 측정소 비교 결과를 통합
                final_abnormal_probability = (
                    abnormal_probability * 0.5) + ((1 - avg_confidence) * 50)
                final_normal_probability = (
                    normal_probability * 0.5) + (avg_confidence * 50)

        station_section += f"\n### 종합 판정 확률 (연관 측정소 및 기존 판정 결과 반영) ###\n"
        # station_section += f"- 정상 확률: {final_normal_probability:.2f}%\n"
        station_section += f"\n- 베이스라인 이상 확률: {final_abnormal_probability:.2f}%\n"

        prompt = (
            "## 데이터 판정 요청 ##\n"
            f"다음은 측정소 {original_data['region']} 에서 {original_data['start_time']} 부터 {original_data['end_time']} 까지 수집된 {original_data['element']} 성분의 비교 결과입니다.\n\n"
            "### 유사한 기존 판정 결과 ###\n"
            f"{similar_results_text}\n\n"
            # "### 유사한 기존 판정결과의 가중 평균을 이용한 베이스라인 이상 확률 분석 ###\n"
            # f"- 정상 확률: {normal_probability:.2f}%\n"
            # f"- 베이스라인 이상 확률: {abnormal_probability:.2f}%\n\n"
            f"{extreme_difference_comment}"
            f"### 연관 측정소 비교 결과 ###\n{station_section}\n"
            "### 요청 사항 ###\n"
            "위 데이터를 종합적으로 분석하여, 해당 데이터가 정상인지 베이스라인 이상인지 판정하고 이유를 설명해주세요.\n"
            "연관 측정소 비교 결과와 유사한 기존 판정 결과의 지역번호, 성분, 상태, 유사도, 측정일시를 사용자에게 보여주세요\n"
            # "답변의 첫 단어는 판정 결과인 '정상' 또는 '이상'이어야 합니다.\n"
        )

        print("###BUILD_EXPLAIN_PROMPT### prompt: ", prompt)
        return prompt

    def explain_query(self, original_data: dict, station_comparison_results: List[dict], similar_data: dict) -> str:
        """
        Step 6: Explain the query with the augmented response using LLM.

        Args:
            results (dict): The results of the query.
            original_data (dict): The original time-series data.
            station_comparison_results (List[dict]): The comparison results for stations.
            element_comparison_results (List[dict]): The comparison results for elements 

        Returns:
            str: The explanation generated by LLM.
        """
        print("###EXPLAIN_QUERY###")
        # LLM 프롬프트 생성
        llm_prompt = self.build_explain_prompt(
            original_data, station_comparison_results, similar_data)

        '''
        messages = [
            {"role": "system", "content": "당신은 사용자의 질문을 JSON 응답 형식으로 변환하는 AI입니다. 반드시 JSON 형식으로만 답변하세요."},
            {"role": "user", "content": llm_prompt}
        ]

        try:
            response = call_ollama_chat_api(
                ollama_host=self.valves.OLLAMA_HOST,
                messages=messages,
                model=self.valves.TEXT_TO_TS_TO_EMBEDDING_MODEL,
                temperature=0.7
            )

            # LLM 응답을 JSON으로 변환
            # TODO: Implement following line with refactored code
            parsed_response = parse_llm_response(self.valves.OLLAMA_HOST, response)

            return parsed_response if parsed_response else response

        except Exception as e:
            return f"LLM 호출 중 오류 발생: {e}"
        '''
        messages = [
            {"role": "system", "content": "당신은 시계열데이터 이상판정 도우미 입니다. 질문에 대해 명확하고 정확한 답변을 해주세요"},
            {"role": "user", "content": llm_prompt}
        ]

        try:
            response = call_ollama_chat_api(
                ollama_host=self.valves.OLLAMA_HOST,
                messages=messages,
                model=self.valves.TEXT_TO_TS_TO_EMBEDDING_MODEL,
                temperature=0.7
            )
            return response

        except Exception as e:
            return f"LLM 호출 중 오류 발생: {e}"

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Generator[str, None, None]:
        """
        Main pipeline execution for both time-series and general queries.
        Outputs results in a streaming fashion using a generator.
        """
        print("###PIPE### user_messages: ", user_message)

        try:
            # Step 1: Parse user input
            query = parse_query(ollama_host=self.valves.OLLAMA_HOST,
                                model=self.valves.TEXT_TO_TS_TO_EMBEDDING_MODEL,
                                user_input=user_message)

            # Check if `query` is a plain string (not a dictionary)

            if isinstance(query, dict) and query.get("type") == "error":
                yield f"오류 발생: {query['message']}"
                return

            if isinstance(query, str):
                yield from query_general_question(ollama_host=self.valves.OLLAMA_HOST,
                                                  model=self.valves.TEXT_TO_TS_TO_EMBEDDING_MODEL,
                                                  question=query)
                return

            validate_query(query)

            # Step 2: Handle time-series queries
            if query["type"] == "time_series":
                # Fetch data from CSV or DB
                original_data = fetch_data(self.valves.POSTGRESQL_USER,
                                           self.valves.POSTGRESQL_PASSWORD,
                                           self.valves.POSTGRESQL_URL,
                                           self.valves.POSTGRESQL_PORT,
                                           self.valves.POSTGRESQL_DB,
                                           self.valves.DOUBLE_THE_SEQUENCE,
                                           self.valves.ADDITIONAL_DAYS,
                                           query,
                                           self.db_path)
                print("###PIPE### original_data: ", original_data)

                # Check if original_data contains valid values
                if not original_data.get("values"):
                    yield f"요청하신 시간 범위에 대해 {query['element']} 데이터가 존재하지 않습니다."
                    return

                related_stations = self.station_network.get_related_station(
                    query['region'], query['element'])

                print(
                    f"###PIPE### Station Related To {query['region']} type {type(query['region'])}: ", related_stations)

                related_station_data = [
                    fetch_data(
                        self.valves.POSTGRESQL_USER,
                        self.valves.POSTGRESQL_PASSWORD,
                        self.valves.POSTGRESQL_URL,
                        self.valves.POSTGRESQL_PORT,
                        self.valves.POSTGRESQL_DB,
                        self.valves.DOUBLE_THE_SEQUENCE,
                        self.valves.ADDITIONAL_DAYS,
                        # Convert element type from numpy.int32 to int
                        {**query, 'region': int(station)},
                        self.db_path)
                    for station in related_stations
                ]

                # Compare the time-series data using FastDTW
                station_comparison_results = compare_time_series(
                    original_data, related_station_data, "station")

                # Embed the time-series values
                embedding = self.embed_values(original_data["values"])

                # Query ChromaDB for similar data
                similar_data = query_chromadb_with_filter(
                    vector_db_host=self.valves.VECTOR_DB_HOST,
                    vector_db_port=self.valves.VECTOR_DB_PORT,
                    collection_id=self.collection_id,
                    embedding=embedding,
                    target_element=query['element'],
                    vector_db_top_k=self.valves.VECTOR_DB_TOP_K)

                # Convert ChromaDB results to a natural language response
                yield from self.explain_query(
                    original_data,
                    station_comparison_results,
                    similar_data
                )

            # Step 3: Handle general queries
            elif query["type"] == "general":
                yield from query_general_question(ollama_host=self.valves.OLLAMA_HOST,
                                                  model=self.valves.TEXT_TO_TS_TO_EMBEDDING_MODEL,
                                                  question=query["question"])
            else:
                yield "알 수 없는 쿼리 유형입니다. 다시 시도해주세요."

        except ValueError as ve:
            yield f"입력 검증 중 오류가 발생했습니다: {str(ve)}"
        except Exception as e:
            # Handle unexpected errors
            yield f"처리 중 오류가 발생했습니다: {str(e)}"
