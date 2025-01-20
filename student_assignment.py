import requests
import base64

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_core.chat_history import InMemoryChatMessageHistory
from PIL import Image
from io import BytesIO

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

history = {}

def getLLM():
    return AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )

def getCalendarificData(year, country, month):
    calendarificURL = "https://calendarific.com/api/v2/holidays?&api_key=z1DU2ThYZiU65id9DGvtiMpkb4XS16TK"
    response = requests.get(calendarificURL+"&country="+country+"&year="+ year+"&month="+month)
    return response.json()

def get_by_session_id(session_id):
    if(session_id not in history):
        history[session_id] = InMemoryChatMessageHistory()
    return history[session_id]

def generate_hw01(question):
    llm = getLLM()
    prompt = " 依據問句的語言回答對應的語系，僅使用正確的Json格式回應 ,格式範例如下：{\"Result\": [{\"date\": \"2024-10-10\",\"name\": \"國慶日\"}]}"
    message = HumanMessage(
            content=[
                {"type": "text", "text": question+prompt},
            ]
    )
    response = llm.invoke([message])
    return JsonOutputParser().invoke(response)
    
def generate_hw02(question):
    llm = getLLM()
    prompt = " 請解析出此問句中的國家名、年、月，並將國家名轉成country code，年、月轉成數字格式，並回傳Json格式，最外層key為\'Result\'，內部key為\'year\'、\'month\'、\'country\'"
    message = HumanMessage(
            content=[
                {"type": "text", "text": question+prompt},
            ]
    )
    response = llm.invoke([message])
    json_response = JsonOutputParser().invoke(response)
    calendar_response = getCalendarificData(str(int(json_response['Result']['year'])), json_response['Result']['country'], str(int(json_response['Result']['month'])))
    prompt = "parse下列JSON檔案，取出holidays.name和holidays.date.iso，轉成JOSN，格式範例如下:{\"Result\": [{\"date\": \"2024-10-10\",\"name\": \"國慶日\"},{\"date\": \"2024-10-09\",\"name\": \"重陽節\"}]}。請直接告訴我結果，無須告訴我過程。愈解析JSON如下  "
    message = HumanMessage(
            content=[
                {"type": "text", "text": prompt + str(calendar_response)},
            ]
    )
    response = llm.invoke([message])
    return JsonOutputParser().invoke(response)

def generate_hw03(question2, question3):
    llm = getLLM()
    
    # 問題解析的提示
    prompt = " 請解析出此問句中的國家名、年、月，並將國家名轉成country code，年、月轉成數字格式，並回傳Json格式，最外層key為'Result'，內部key為'year'、'month'、'country'"
    
    # 構建HumanMessage，發送問題給LLM
    message = HumanMessage(content=question2 + prompt)
    
    # 使用 `RunnableLambda` 封装消息
    message_runnable = RunnableLambda(lambda _: message)
    
    # 使用 `RunnableWithMessageHistory` 記錄歷史
    llm_with_history = RunnableWithMessageHistory(llm, get_by_session_id, input_messages_key='', history_messages_key='')

    # 創建RunnableSequence時，將Runnable物件逐個傳遞給構造函數
    runnable_sequence = RunnableSequence(
        message_runnable,
        llm_with_history
    )

    # 執行並獲得LLM的回應
    response = runnable_sequence.invoke({'':'foo'}, {'configurable': {'session_id': 'bob'}}).content
    
    # 解析LLM返回的JSON結果
    try:
        json_response = JsonOutputParser().invoke(response)  # 假設response是有效的JSON格式
        year = str(int(json_response['Result']['year']))
        month = str(int(json_response['Result']['month']))
        country = json_response['Result']['country']
    except Exception as e:
        return {"error": "無法解析JSON結果", "message": str(response)}

    # 從Calendarific API獲取假期數據
    calendar_response = getCalendarificData(year, country, month)
    
    # 構建新的HumanMessage來處理日曆數據
    message = HumanMessage(content=prompt + str(calendar_response))
    
    # 轉換message為Runnable
    message_runnable = RunnableLambda(lambda _: message)

    # 再次構建 `RunnableSequence` 串聯 `message_runnable` 和 llm_with_history
    runnable_sequence = RunnableSequence(
        message_runnable,
        llm_with_history
    )
    
    # 再次執行並獲得最終回應
    second_response =  JsonOutputParser().invoke(runnable_sequence.invoke({'':'foo'}, {'configurable': {'session_id': 'bob'}}).content)
    
    # 判斷是否存在之前的記憶內
    
    final_prompt = ' 請用json格式回應，[Result][add]為bool，判斷是否需加入, [Result][reason]為原因。並用中文回答'
    # 構建HumanMessage，發送問題給LLM
    message = HumanMessage(content=question3 + final_prompt)
    
    # 使用 `RunnableLambda` 封装消息
    message_runnable = RunnableLambda(lambda _: message)
    
    # 使用 `RunnableWithMessageHistory` 記錄歷史
    llm_with_history = RunnableWithMessageHistory(llm, get_by_session_id, input_messages_key='', history_messages_key='')

    # 創建RunnableSequence時，將Runnable物件逐個傳遞給構造函數
    runnable_sequence = RunnableSequence(
        message_runnable,
        llm_with_history
    )

    # 執行並獲得LLM的回應
    final_response = JsonOutputParser().invoke(runnable_sequence.invoke({'':'foo'}, {'configurable': {'session_id': 'bob'}}).content)

    return final_response

def readImage(image_path):
    with open(image_path, 'rb') as image_file:
         image_bytes = image_file.read()
    image_base64  = base64.b64encode(image_bytes).decode("utf-8")
    return image_base64 

def generate_hw04(question):
    llm = getLLM()
    
    # 將圖像數據編碼為Base64
    image_data = readImage('baseball.png')

    # 構建消息
    prompt = "  請用JSON格式輸出，{'Result': {'score': 124}} "
    message = HumanMessage(
        content=[
            {"type": "text", "text": question + prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
        ]
    )

    # 調用LLM
    response = llm.invoke([message])
    return JsonOutputParser().invoke(response)

def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    return response

print(generate_hw04("請問中華台北的積分是多少?"))