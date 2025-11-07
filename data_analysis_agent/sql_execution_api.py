from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import mysql.connector
from mysql.connector import Error
from typing import List, Dict, Any
import logging
import os

app = FastAPI(title="SQL Execution API")

MYSQL_CONFIG = {
    'host': os.getenv('MYSQL_HOST', 'localhost'),
    'port': os.getenv('MYSQL_PORT', 3306),
    'user': os.getenv('MYSQL_USER', 'root'),
    'password': os.getenv('MYSQL_PASSWORD', '123456'),
    'database': os.getenv('MYSQL_DATABASE', 'ecommerce_platform')
}

class SQLRequest(BaseModel):
    sql: str

class SQLResponse(BaseModel):
    success: bool
    data: List[Dict[str, Any]] = []
    error: str = ""
    row_count: int = 0

def get_mysql_connection():
    """获取MySQL数据库连接"""
    try:
        connection = mysql.connector.connect(**MYSQL_CONFIG)
        return connection
    except Error as e:
        logging.error(f"MySQL connection error: {e}")
        raise


def execute_sql_query(sql: str) -> SQLResponse:
    """执行SQL查询并返回结果"""
    connection = None
    try:
        connection = get_mysql_connection()
        cursor = connection.cursor(dictionary=True)  # 返回字典格式

        # 安全检查：防止危险操作（根据您的需求调整）
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE']
        sql_upper = sql.upper().strip()

        # 允许SELECT查询，对其他操作进行限制
        # if not sql_upper.startswith('SELECT'):
        #     return SQLResponse(
        #         success=False,
        #         error="Only SELECT queries are allowed for security reasons"
        #     )

        # 执行查询
        cursor.execute(sql)

        # 获取结果
        if sql_upper.startswith('SELECT'):
            results = cursor.fetchall()
            data = results
            row_count = len(data)
        else:
            connection.commit()
            data = []
            row_count = cursor.rowcount

        return SQLResponse(
            success=True,
            data=data,
            row_count=row_count
        )

    except Error as e:
        logging.error(f"MySQL execution error: {e}")
        return SQLResponse(
            success=False,
            error=str(e)
        )
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return SQLResponse(
            success=False,
            error=str(e)
        )
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()


@app.post("/execute-sql", response_model=SQLResponse)
async def execute_sql(request: SQLRequest):
    """执行SQL查询的接口"""
    return execute_sql_query(request.sql)

if __name__ == "__main__":
    uvicorn.run("sql_execution_api:app", host="0.0.0.0", port=8000, reload=True)