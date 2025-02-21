from datetime import datetime
from typing import Union
import json
import re

from NIERModules.ollama_handler.ollama_handler import call_ollama_generate_api


def parse_query(ollama_host:str, model:str,user_input: str) -> Union[dict, str]:
    """
    Parse user input into a structured query format.

    Args:
        ollama_host (str): URL of the Ollama API server.
        model (str): The name of the LLM to use.
        user_input (str): The user's query in natural language.

    Returns:
        Union[dict, str]: A structured query dictionary or the original user input.
    """    
    
    print("###PARSE_QUERY### user_input:", user_input)

    ts_pattern = re.compile(
        r"^(?P<region>\d{6}),\s*"                   # 6-digit region
        r"(?P<element>NO|NO2|NOX|PM10|PM25|SO2|CO|O3),\s*"  # Element (uppercase)
        r"(?P<start>\d{10}),\s*"                     # YYYYMMDDHH
        r"(?P<end>\d{10})\s*$",                      # YYYYMMDDHH
        re.IGNORECASE  
    )
    match = ts_pattern.match(user_input)
    
    if match:
        # Time-series query detected via Regex
        def format_time(ymdh: str) -> str:
            return datetime.strptime(ymdh, "%Y%m%d%H").strftime("%Y-%m-%d %H:%M:%S")

        start_time = format_time(match.group("start"))
        end_time = format_time(match.group("end"))

        return {
            "type": "time_series",
            "region": int(match.group("region")),
            "element": match.group("element"),
            "start_time": start_time,
            "end_time": end_time
        }
    # Step 1-2: Use Ollama API for Few-shot Prompt if Regex fails
    try:
        print("###PARSE_QUERY### Parsing User Input with Ollama API.")
        response = parse_with_ollama(
            ollama_host=ollama_host, 
            model=model,
            user_input=user_input)
        
        print("###PARSE_QUERY### Ollama response:", response, type(response))
        # If Ollama returns a structured dict, pass it on
        if isinstance(response, dict) and "type" in response:
            return response
        
    except Exception as e:
        # Log the error and fallback to treating input as plain text
        print(f"###PARSE_QUERY### Error during Ollama parsing: {e}")
    # Fallback: Treat input as a plain text response
    print("###PARSE_QUERY### Fallback to plain text response.")
    return user_input


def parse_with_ollama(ollama_host:str, model:str, user_input: str) -> Union[dict, None]:
    """
    Parse user input into JSON format using Ollama with explicit schema.
    !!! WARNING !!! Requires Ollma 0.5.0 or higher.
    Args:
        user_input (str): The user's query in natural language.
    Returns:
        Union[dict, None]: Parsed JSON response or None on failure.
    """
    
    current_YYYY_MM_DD = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    prompt_template = """
    당신은 사용자가 입력한 자연어 형식의 쿼리를 JSON 형식으로 변환합니다.
    반드시 JSON 형식으로만 답변하세요.
    
    ## 지침:
    - 사용자의 질문이 특정 시간 범위와 지역에 대한 데이터 요청일 경우:
        - `type`: "time_series"
        - `region`: 지역 코드 (숫자)
        - `start_time`: 시작 시간 (YYYY-MM-DD HH:00:00)
        - `end_time`: 종료 시간 (YYYY-MM-DD HH:00:00)
        - `element`: 요청된 성분명 (대문자)
    - 사용자의 질문이 일반적인 자연어 질문일 경우:
        - `type`: "general"
        - `question`: 질문 내용을 그대로 포함
    - 주의사항:
        - JSON 이외의 형식을 반환하지 마세요.
        - 예시를 그대로 복사해서는 안 됩니다.
        - 답변은 사용자 입력에 맞게 생성되어야 합니다.
        - **시간(`HH`)은 01시부터 24시까지의 범위만이 존재하며, 분(`MM`)과 초(`SS`)는 항상 `00`이어야 합니다.**
        - **다음 성분들만 유효합니다:**  
          `NO`, `NO2`, `NOX`, `PM10`, `PM25`, `SO2`, `CO`, `O3`
    - 추가 정보:
        - 오늘 날짜는 {current_YYYY_MM_DD}입니다.
    ## 예시:
    
    
    질문: 437401 지역의 CO성분 22년 1월 27일 18시 부터 01월 28일 14시 까지 성분을 조회할게
    답변: {{"type": "time_series", "region": 437401, "start_time": "2022-01-27 18:00:00", "end_time": "2022-01-28 14:00:00", "element": "CO"}}
    질문: 111123 측정소 NO2  20220803 18시 ~ 20220910 14시 데이터에 문제가 있나?
    답변: {{"type": "time_series", "region": 111123, "start_time": "2022-08-03 18:00:00", "end_time": "2022-09-10 14:00:00", "element": "NO2"}}
    
    질문: 24년 1월 25일 733301 측정소 PM2.5 성분 조회
    답변: {{"type": "time_series", "region": 733301, "start_time": "2024-01-25 01:00:00", "end_time": "2024-01-25 24:00:00", "element": "PM25"}}
    
    질문: 이번 주 서울의 미세먼지 농도는 어떻게 되나요?
    답변: {{"type": "general", "question": "이번 주 서울의 미세먼지 농도는 어떻게 되나요?"}}
    
    질문: 이산화탄소가 무엇인가요?
    답변: {{"type": "general", "question": "이산화탄소가 무엇인가요?"}}
    
    
    ## 사용자 입력:
    질문: {user_input}
    답변:
    """
    
    prompt = prompt_template.format(
        user_input=user_input, current_YYYY_MM_DD=current_YYYY_MM_DD
    )
    
    # TODO: use Pydantic (recommended) to serialize the schema using model_json_schema().
    # https://ollama.com/blog/structured-outputs
    # JSON
    json_schema = {
        "type": "object",
        "properties": {
            "type": {"type": "string"},
            "region": {"type": "integer"},
            "start_time": {"type": "string"},
            "end_time": {"type": "string"},
            "element": {"type": "string"},
            "question": {"type": "string"}
        },
        "required": ["type"]
    }
    try:
        response = call_ollama_generate_api(
            ollama_host = ollama_host,
            prompt=prompt,
            model=model,
            temperature=0.0,
            stream=False,
            format=json_schema
        )
        if isinstance(response, dict) and "response" in response:
            try:
                parsed_response = json.loads(response["response"])
                if isinstance(parsed_response, dict) and "type" in parsed_response:
                    return parsed_response
            except json.JSONDecodeError as e:
                print(f"###ERROR### JSON 파싱 오류: {e}")
                return None
        if isinstance(response, dict) and "type" in response:
            return response
        else:
            raise ValueError("Invalid response format from Ollama API.")
    except Exception as e:
        print(f"###ERROR### Ollama API 호출 중 오류 발생: {e}")
        return None
        

def validate_query(query: dict):
    """
    Validates the query dictionary to ensure it contains the required keys
    and the values meet the expected types or formats.
    
    Args:
        query (dict): The query dictionary to validate.
        
    Raises:
        ValueError: If the query is missing required keys or contains invalid values.
    """
    
    required_keys = ["type"]  # All queries must have a type
    
    if query["type"] == "time_series":
        required_keys += ["region", "start_time", "end_time", "element"]
        
    elif query["type"] == "general":
        required_keys += ["question"]
        
    # Check if all required keys are present
    for key in required_keys:
        if key not in query:
            raise ValueError(f"Missing required key: {key}")
        
    # Additional type and format validations for time_series
    if query["type"] == "time_series":
        if not isinstance(query["region"], int):
            raise ValueError(
                f"Invalid type for 'region': Expected int, got {type(query['region'])}")
        
        if not isinstance(query["start_time"], str) or not isinstance(query["end_time"], str):
            raise ValueError(
                "Invalid type for 'start_time' or 'end_time': Expected str")
        
        if not isinstance(query["element"], str):
            raise ValueError(
                f"Invalid type for 'element': Expected str, got {type(query['element'])}")
   
    # Additional validation for general queries
    if query["type"] == "general" and not isinstance(query["question"], str):
        raise ValueError(
            f"Invalid type for 'question': Expected str, got {type(query['question'])}")
