import requests
from typing import List, Union
import json


def call_ollama_generate_api(ollama_host:str, prompt: str, model: str, temperature: float = 0.7, stream: bool = False, format: Union[str, dict] = "text", retries: int = 3) -> Union[dict, str]:
    """
    Calls the Ollama API with the specified prompt and model.
    
    Args:
        prompt (str): The prompt to send to the API.
        model (str): The model name to use for the API call.
        temperature (float): The temperature to control response randomness.
        stream (bool): Whether to enable streaming output.
        format (str): Expected response format ("json" or "text").
        retries (int): Number of retries for the API call.
        
    Returns:
        Union[dict, str]: The API response as a parsed JSON or plain text.
    """
    url = f"{ollama_host}/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "stream": stream,
    }
    
    # Check if format is JSON schema
    if isinstance(format, dict):
        payload["format"] = format
    elif format == "json":
        payload["format"] = "json"
    for attempt in range(retries):
        try:
            response = requests.post(
                url, json=payload, headers=headers, stream=stream)
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "") if format == "text" else data
            else:
                print(
                    f"###API CALL FAILED### Status Code: {response.status_code}, Response: {response.text}")
                if attempt < retries - 1:
                    print(f"Retrying... ({attempt + 1}/{retries})")
        except requests.RequestException as e:
            print(f"###REQUEST EXCEPTION### {e}")
            if attempt < retries - 1:
                print(f"Retrying... ({attempt + 1}/{retries})")
   
    # If all retries fail
    raise RuntimeError(f"Ollama API 호출 실패 after {retries} attempts.")


def call_ollama_chat_api(ollama_host:str, messages: List[dict], model: str, temperature: float = 0.7, format: Union[str, dict] = "text") -> Union[dict, str]:
    """
    Calls the Ollama Chat API with the specified messages and model.
    Args:
        messages (List[dict]): The conversation history or prompts.
        model (str): The model name to use for the API call.
        temperature (float): The temperature to control response randomness.
        format (str|dict): The expected response format ("text", "json", or JSON schema).
    -
    Returns:
        Union[dict, str]: The API response as a JSON object or plain text.
    """
    url = f"{ollama_host}/api/chat"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False
    }
    if isinstance(format, dict) or format == "json":
        payload["format"] = format
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data.get("message", {}).get("content", "").strip()
        else:
            raise RuntimeError(
                f"API 호출 실패: {response.status_code} - {response.text}")
    except requests.RequestException as e:
        raise RuntimeError(f"요청 처리 중 오류 발생: {e}")
    
    
def parse_llm_response(ollama_host:str, model:str, llm_response: str) -> Union[dict, None]:
    """
    Parse LLM response into structured JSON format using Ollama API.
    Args:
        llm_response (str): The raw response from LLM.
    Returns:
        Union[dict, None]: Parsed JSON response or None on failure.
    """
    
    
    prompt_template = """
    당신은 사용자가 입력한 텍스트를 JSON 형식으로 변환하는 AI입니다.
    반드시 JSON 형식으로만 답변하세요.
    ## JSON 출력 형식:
    ```json
    {{
      "result": "정상" | "이상",
      "explain": "이 데이터를 판정한 이유를 간결하게 설명합니다.",
      "near_station": [
        {{
          "rank": 1,
          "station_id": "지역번호",
          "similarity": "정상 범위 | 약간 낮음 | 평균보다 낮음 | 강한 이상 징후"
        }},
        ...
      ],
      "previous_results": [
        {{
          "previous_station_id": "지역번호",
          "element": "SO2",
          "result": "정상 | 이상",
          "similarity": "유사 | 다름 | 매우 다름",
          "measurement_period": "YYYY-MM-DD HH:MM:SS ~ YYYY-MM-DD HH:MM:SS"
        }},
        ...
      ]
    }}
    ```
    ## 주의 사항:
    - JSON 이외의 형식을 반환하지 마세요.
    - 판정(`판정`) 필드는 반드시 `"정상"` 또는 `"이상"`이어야 합니다.
    - 추가적인 설명이나 문장은 포함하지 마세요.
    ## 입력 데이터:
    ```
    {llm_response}
    ```
    """
    
    json_schema = {
        "type": "object",
        "properties": {
            "result": {"type": "string", "enum": ["정상", "이상"]},
            "explain": {"type": "string"},
            "near_station": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "rank": {"type": "integer"},
                        "station_id": {"type": "string"},
                        "similarity": {"type": "string"},
                    },
                    "required": ["rank", "station_id", "similarity"],
                },
            },
            "previous_results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "previous_station_id": {"type": "string"},
                        "element": {"type": "string"},
                        "result": {"type": "string"},
                        "similarity": {"type": "string"},
                        "measurement_period": {"type": "string"},
                    },
                    "required": ["previous_station_id", "element", "result", "similarity", "measurement_period"],
                },
            },
        },
        "required": ["result", "explain", "near_station", "previous_results"],
    }
    
    try:
        response = call_ollama_generate_api(
            ollama_host=ollama_host,
            prompt=prompt_template.format(llm_response=llm_response),
            model=model,
            temperature=0.0,
            stream=False,
            format=json_schema,
        )
        if isinstance(response, dict) and "response" in response:
            try:
                parsed_response = json.loads(response["response"])
                if isinstance(parsed_response, dict) and "판정" in parsed_response:
                    return parsed_response
            except json.JSONDecodeError as e:
                print(f"###ERROR### JSON 파싱 오류: {e}")
                return None
        return response if isinstance(response, dict) and "판정" in response else None
    
    except Exception as e:
        print(f"###ERROR### Ollama API 호출 중 오류 발생: {e}")
        return None
    
    
    
def query_general_question(ollama_host:str, model:str, question: str) -> str:
    """
    Handles general, non-time-series questions using Ollama Chat API.   
    
    Args:
        question (str): The user's question.    
    
    Returns:
        str: The response from the LLM.
    """
    
    print("###QUERY_GENERAL_QUESTION###")
    
    messages = [
        {"role": "system", "content": "당신은 사용자 질문에 간결하고 정확한 답변을 제공하는 유용한 도우미입니다."},
        {"role": "user", "content": question}
    ]
    try:
        response = call_ollama_chat_api(
            ollama_host=ollama_host,
            messages=messages,
            model=model,
            temperature=0.7
        )
        if isinstance(response, str):
            print("respones:", response)
            return response.strip()
        else:
            return "API 응답에서 텍스트를 가져오지 못했습니다."
    
    except Exception as e:
        return f"LLM 호출 중 오류 발생: {e}"
        