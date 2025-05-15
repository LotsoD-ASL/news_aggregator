from typing import List, Dict, Any
import requests
from fastmcp import FastMCP
from bs4 import BeautifulSoup

mcp = FastMCP("news_aggregator")

BASE_URL = "https://search.sina.com.cn/?c=news&q={keyword}&range=all&num={limit}&page=1"

@mcp.tool()
async def search_news(keyword: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    关键词新闻搜索（新浪新闻），返回新闻标题、链接、摘要
    """
    url = BASE_URL.format(keyword=keyword, limit=limit)
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    for item in soup.select(".box-result")[:limit]:
        title_tag = item.select_one("h2 a")
        desc_tag = item.select_one(".content")
        if title_tag:
            results.append({
                "title": title_tag.get_text(strip=True),
                "url": title_tag.get("href"),
                "description": desc_tag.get_text(strip=True) if desc_tag else ""
            })
    return results

@mcp.tool()
async def get_news_detail(url: str) -> Dict[str, Any]:
    """
    获取新闻详情正文（简单爬取网页正文）
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(resp.text, "html.parser")
    # 新浪新闻正文常见class
    content = ""
    for cls in ["article", "article-content", "main-content", "content" ]:
        node = soup.find(class_=cls)
        if node:
            content = node.get_text(separator="\n", strip=True)
            break
    if not content:
        # 兜底：取所有p标签文本
        paragraphs = soup.find_all("p")
        content = "\n".join([p.get_text() for p in paragraphs if len(p.get_text()) > 30])
    return {
        "url": url,
        "content": content[:2000]  # 限制长度
    }

@mcp.tool()
async def summarize_text(text: str) -> str:
    """
    对输入文本进行智能摘要（简单版）
    """
    if len(text) < 200:
        return text
    sentences = text.split("。")
    summary = "。".join(sentences[:3]) + "。"
    return summary

@mcp.tool()
async def analyze_sentiment(text: str) -> str:
    """
    对输入文本进行情感分析（简单规则版）
    """
    pos_words = ["好", "积极", "乐观", "增长", "创新", "突破"]
    neg_words = ["差", "下滑", "危机", "负面", "裁员", "亏损"]
    pos = sum(w in text for w in pos_words)
    neg = sum(w in text for w in neg_words)
    if pos > neg:
        return "正面"
    elif neg > pos:
        return "负面"
    else:
        return "中性"

if __name__ == "__main__":
    print("启动新闻聚合MCP服务器...")
    mcp.run(transport='stdio') 