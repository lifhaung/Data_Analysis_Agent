import os
import requests
from langchain.tools import tool
from langchain_deepseek import ChatDeepSeek
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.agents.middleware import SummarizationMiddleware
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver


#prompt
prompt = (
    "你是一个专业的电商数据分析助手，拥有完整的知识体系。你可以综合运用以下所有知识库来为用户提供最佳答案：\n\n"
    "可用知识库：\n"
    "1. 行业知识库 - 电商平台模式、业务流程、关键指标等基础概念\n"
    "2. 数据分析案例库 - 用户行为分析、销售预测、营销效果评估等实战案例\n"
    "3. SQL语法库 - SQL查询技巧、多表关联、高级分析函数等技术方法\n"
    "4. 数据库信息库 - 表结构、字段含义、数据关系等底层信息\n\n"
    "综合回答策略：\n"
    "- 对于复杂的数据分析问题，先理解业务背景（行业知识库），再参考分析方法（案例库），最后考虑技术实现（SQL语法库）\n"
    "- 当用户提出具体业务问题时，结合行业知识和数据分析方法给出全面建议\n"
    "- 在执行SQL查询前，先确认表结构（数据库信息库）确保查询正确\n"
    "- 将理论概念、实战案例和技术实现有机结合，提供立体的解决方案\n\n"
    "工具使用原则：\n"
    "- 主动检索多个相关知识库，构建完整的知识体系\n"
    "- 不要局限于单一知识库，要交叉引用和综合运用\n"
    "- 对于复杂问题，可以按需调用多个工具来获取全面信息\n"
    "- 基于所有检索到的信息，给出综合性的专业建议\n\n"
    "SQL执行规范：\n"
    "- 在执行查询前，确保理解业务需求和数据结构\n"
    "- 生成的SQL要基于正确的表结构和业务逻辑\n"
    "- 对查询结果要结合业务背景进行深度分析\n\n"
    "回答要求：\n"
    "- 融合行业知识、分析方法和技术实现三个层面\n"
    "- 提供从理论到实践的全链路解决方案\n"
    "- 用具体案例和数据支撑你的分析结论\n"
    "- 给出可落地的实操建议\n\n"
    "记住：你的优势在于能够整合所有知识库，提供全方位、多角度的专业分析！"
)


#嵌入模型和向量存储
embeddings = OpenAIEmbeddings(model="Qwen/Qwen3-Embedding-8B")
industry_vector_store = InMemoryVectorStore(embeddings)
data_analysis_vector_store = InMemoryVectorStore(embeddings)
sql_vector_store = InMemoryVectorStore(embeddings)
database_vector_store = InMemoryVectorStore(embeddings)

#indexing
# 定义知识库文件路径
knowledge_files = {
    "industry": "C:\\Users\\14396\\Desktop\\知识库\\行业知识库.pdf",
    "data_analysis": "C:\\Users\\14396\\Desktop\\知识库\\数据分析案例库.pdf",
    "sql": "C:\\Users\\14396\\Desktop\\知识库\\SQL语法库.pdf",
    "database": "C:\\Users\\14396\\Desktop\\知识库\\数据库表格信息.pdf"
}

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=20,
    add_start_index=True
)

# 分别存储每个知识库
for knowledge_type, file_path in knowledge_files.items():
    try:
        print(f"正在存储 {knowledge_type} 知识库...")

        # 加载文档
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # 文本分割
        splits = text_splitter.split_documents(docs)

        # 存储到对应的向量库
        if knowledge_type == "industry":
            industry_vector_store.add_documents(documents=splits)
        elif knowledge_type == "data_analysis":
            data_analysis_vector_store.add_documents(documents=splits)
        elif knowledge_type == "sql":
            sql_vector_store.add_documents(documents=splits)
        elif knowledge_type == "database":
            database_vector_store.add_documents(documents=splits)

        print(f"{knowledge_type} 知识库存储完成: {len(splits)} 个片段")
    except Exception as e:
        print(f"{knowledge_type} 知识库存储失败: {e}")

#llm
advanced_model = ChatDeepSeek(
    model="deepseek-reasoner",
    temperature=0.7
)
basic_model= ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.7
)


# rag agent
@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """结合对话历史和问题复杂度选择模型"""

    messages = request.state["messages"]
    last_message = messages[-1]
    user_query = last_message.text.lower() if hasattr(last_message, 'text') else ""

    # 1. 基于关键词判断
    reasoning_keywords = ['分析', '推理', '为什么', '如何', '策略', '优化', '预测', '评估']
    simple_keywords = ['查询', '显示', '列出', '数量', '什么是']

    has_reasoning = any(keyword in user_query for keyword in reasoning_keywords)
    has_simple = any(keyword in user_query for keyword in simple_keywords)

    # 2. 基于问题长度判断
    query_length = len(user_query)
    is_long_query = query_length > 20

    # 3. 基于对话轮数判断
    conversation_turns = len([msg for msg in messages if hasattr(msg, 'role') and msg.role == 'user'])

    # 决策矩阵
    if has_reasoning or is_long_query or conversation_turns > 3:
        print(f"使用高级模型 (推理关键词: {has_reasoning}, 长度: {query_length}, 轮数: {conversation_turns})")
        model = advanced_model
    elif has_simple and query_length < 50:
        print(f"使用基础模型 (简单查询)")
        model = basic_model
    else:
        print(f"使用基础模型 (默认)")
        model = basic_model

    request.model = model
    return handler(request)


@tool(response_format="content_and_artifact")
def retrieve_industry_knowledge(query: str):
    """检索电商行业知识，包括平台类型、业务流程、关键指标等基础概念。"""
    retrieved_docs = industry_vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: 行业知识库\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

@tool(response_format="content_and_artifact")
def retrieve_data_analysis_cases(query: str):
    """检索数据分析案例，包括用户行为分析、销售预测、营销效果评估等实战案例。"""
    retrieved_docs = data_analysis_vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: 数据分析案例库\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

@tool(response_format="content_and_artifact")
def retrieve_sql_knowledge(query: str):
    """检索SQL语法和查询技巧，包括基础查询、多表关联、高级分析函数等。"""
    retrieved_docs = sql_vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: SQL语法库\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

@tool(response_format="content_and_artifact")
def retrieve_database_info(query: str):
    """检索数据库表结构信息，包括表字段说明、关系模型、数据字典等。"""
    retrieved_docs = database_vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: 数据库表格信息\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


@tool(response_format="content_and_artifact")
def execute_sql_query(sql: str):
    """
    执行SQL查询并返回结果。当用户需要查询数据库信息、数据报表或特定数据时使用此工具。
    你要是不知道数据库中表单长什么样子，先查询需要的表单结构
    功能：
    - 执行安全的SELECT查询语句
    - 返回格式化的查询结果
    - 支持用户、产品、订单等数据查询

    参数：
        sql: 要执行的SQL查询语句

    返回：
        查询结果文本和原始数据对象
    """
    try:
        # FastAPI 后端地址
        api_url = "http://localhost:8000/execute-sql"

        # 发送请求到后端
        response = requests.post(api_url, json={"sql": sql})
        result_data = response.json()

        if result_data["success"]:
            if result_data["data"]:
                # 格式化查询结果
                formatted_data = []
                for row in result_data["data"]:
                    formatted_row = ", ".join([f"{k}: {v}" for k, v in row.items()])
                    formatted_data.append(formatted_row)

                result_text = f"Query executed successfully. Found {result_data['row_count']} rows.\n\nResults:\n" + "\n".join(
                    formatted_data)
            else:
                result_text = f"Query executed successfully. {result_data['row_count']} rows affected."

            return result_text, result_data
        else:
            error_msg = f"SQL execution failed: {result_data['error']}"
            return error_msg, result_data

    except Exception as e:
        error_msg = f"Failed to connect to SQL execution service: {str(e)}"
        return error_msg, {"success": False, "error": str(e)}

sum_middleware = SummarizationMiddleware(
            model="deepseek-chat",
            max_tokens_before_summary=4000,
            messages_to_keep=20,
            summary_prompt="请用中文简洁地总结以下内容的关键要点：",
        )

# human_middleware = HumanInTheLoopMiddleware(
#     interrupt_on={
#         # SQL执行工具需要人工审批（防止危险操作）
#         "execute_sql_query": {
#             "allowed_decisions": ["approve", "edit", "reject"],
#             "reason": "SQL查询可能影响数据库，需要人工确认"
#         },
#         # 知识库检索工具自动批准
#         "retrieve_industry_knowledge": False,
#         "retrieve_data_analysis_cases": False,
#         "retrieve_sql_knowledge": False,
#         "retrieve_database_info": False,
#     }
# )

tools = [retrieve_industry_knowledge, retrieve_data_analysis_cases, retrieve_sql_knowledge, retrieve_database_info,execute_sql_query]
middleware=[sum_middleware]

agent = create_agent(
    model=basic_model,
    tools=tools,
    middleware=middleware,
    # checkpointer=InMemorySaver(),
    system_prompt=prompt
)

#测试
# query = (
#     "查询用户信息"
# )
#
# for event in agent.stream(
#     {"messages": [{"role": "user", "content": query}]},
#     # config={"configurable": {"thread_id": "1"}},
#     stream_mode="values",
# ):
#     event["messages"][-1].pretty_print()
#
#
# def multi_round_chat():
#     """多轮对话函数"""
#     print("电商数据分析助手已启动！输入 '退出' 或 'quit' 结束对话。")
#     print("=" * 50)
#
#     # 初始化对话历史
#     conversation_history = []
#
#     while True:
#         # 获取用户输入
#         user_input = input("您: ").strip()
#
#         # 退出条件
#         if user_input.lower() in ['退出', 'quit', 'exit', 'q']:
#             print("感谢使用，再见！")
#             break
#
#         if not user_input:
#             print("请输入您的问题:")
#             continue
#
#         # 构建消息历史
#         messages = conversation_history + [{"role": "user", "content": user_input}]
#
#         print("AI正在思考...")
#
#         try:
#             # 执行对话
#             for event in agent.stream(
#                     {"messages": messages},
#                     stream_mode="values",
#             ):
#                 event["messages"][-1].pretty_print()
#                 conversation_history = event["messages"]
#
#         except Exception as e:
#             print(f"对话出错: {e}")
#             print("请重新输入您的问题:")
#
#
# # 启动多轮对话
# multi_round_chat()