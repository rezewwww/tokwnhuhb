"""
初始化数据库。首个注册用户自动成为管理员。
运行: python init_db.py
"""
from models import init_db


if __name__ == "__main__":
    print("初始化数据库表结构...")
    init_db()
    print("完成！首个注册用户将自动成为管理员。")
