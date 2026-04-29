from dotenv import load_dotenv
from zai import ZhipuAiClient

load_dotenv()


def zhipu_ocr(image_url: str):
    # 初始化客户端
    client = ZhipuAiClient()
    # 调用布局解析 API
    response = client.layout_parsing.create(model="glm-ocr", file=image_url)

    # 输出结果
    return response.md_results
