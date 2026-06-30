# TokenHub - AI API 中转站 部署指南

## 版本说明

- **源码版（¥2,999）** — 源码 + 本文档，自行部署
- **部署版（¥4,999）** — 源码 + 帮你部署到服务器并调试
- **企业版（¥9,999）** — 源码 + 部署 + Logo/品牌修改 + 真实支付接入

## 快速部署

### 方式一：直接启动（开发测试）

```bash
pip install -r requirements.txt
python app.py
```

浏览器访问 http://localhost:5000

### 方式二：Docker 部署（推荐生产）

```bash
cp .env.example .env
# 编辑 .env 填入你的 API Key
docker compose up -d
```

## 使用流程

1. 打开 http://localhost:5000
2. 注册第一个账号 → 自动成为管理员
3. 左侧菜单 → API Keys → 创建新 Key
4. 左侧菜单 → 充值中心 → 点「确认支付」给自己充值
5. 用 API Key 调用接口：

```bash
curl http://localhost:5000/v1/chat/completions \
  -H "Authorization: Bearer sk-你的key" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "你好"}]}'
```

## Python 调用

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:5000/v1",
    api_key="sk-你的key"
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "你好"}]
)
print(response.choices[0].message.content)
```

## 文件结构

```
├── app.py               # Flask 主程序
├── models.py            # 数据库模型
├── init_db.py           # 数据库初始化
├── index.html           # 前端页面
├── requirements.txt     # Python 依赖
├── Dockerfile           # Docker 构建
├── docker-compose.yml   # Docker 编排
├── .env.example         # 环境变量配置（复制为 .env 后编辑）
├── data.db              # SQLite 数据库
└── README.md            # 本文件
```

## 技术支持

购买部署版/企业版请联系卖家远程协助。
