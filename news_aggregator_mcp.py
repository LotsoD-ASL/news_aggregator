import logging
from typing import List, Dict, Any
from fastmcp import FastMCP
import requests
from bs4 import BeautifulSoup
import os
import hashlib
import json
import time
import jieba
import jieba.analyse
from collections import defaultdict
from typing import Optional
import asyncio
import aiohttp
from datetime import datetime, timedelta
from difflib import SequenceMatcher

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 新闻源配置
NEWS_SOURCES = {
    "sina": "https://search.sina.com.cn/?c=news&q={keyword}&range=all&num={limit}&page=1",
    "sohu": "https://www.sohu.com/search/news?keyword={keyword}&page=1",
    "163": "https://www.163.com/search?keyword={keyword}"
}

# 新闻分类关键词
NEWS_CATEGORIES = {
    "科技": ["人工智能", "互联网", "数字化", "科技", "创新", "技术"],
    "财经": ["股市", "经济", "金融", "投资", "理财", "市场"],
    "社会": ["民生", "社会", "事件", "生活", "教育"],
    "文化": ["文化", "艺术", "娱乐", "电影", "音乐"],
    "体育": ["体育", "运动", "比赛", "足球", "篮球"]
}

# 缓存配置
CACHE_DIR = "news_cache"
CACHE_EXPIRE = 3600  # 缓存过期时间（秒）
CLEANUP_INTERVAL = 3600  # 清理间隔（秒）
CACHE_EXPIRY = 86400    # 缓存过期时间（秒）

# 全局新闻缓存字典
news_cache = {}

# 创建缓存目录
os.makedirs(CACHE_DIR, exist_ok=True)

# 创建MCP实例
mcp = FastMCP("news")

def get_cached_data(key: str) -> Optional[Dict]:
    """从缓存获取数据"""
    cache_file = os.path.join(CACHE_DIR, f"{hashlib.md5(key.encode()).hexdigest()}.json")
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if time.time() - data['timestamp'] < CACHE_EXPIRE:
                return data['content']
    return None

def save_to_cache(key: str, content: Any):
    """保存数据到缓存"""
    cache_file = os.path.join(CACHE_DIR, f"{hashlib.md5(key.encode()).hexdigest()}.json")
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': time.time(),
            'content': content
        }, f, ensure_ascii=False)

def clean_expired_cache():
    """清理过期的缓存"""
    try:
        current_time = time.time()
        expired_keys = [k for k, v in news_cache.items() if current_time - v['timestamp'] > CACHE_EXPIRY]
        for key in expired_keys:
            del news_cache[key]
        if expired_keys:
            logger.debug(f"Cleaned {len(expired_keys)} expired cache entries")
    except Exception as e:
        logger.error(f"Cache cleanup error: {str(e)}")

@mcp.tool("search_news")
async def search_news(keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
    """搜索新闻接口"""
    clean_expired_cache()

    async def fetch_news(source_name: str, url_template: str) -> List[Dict[str, Any]]:
        try:
            url = url_template.format(keyword=keyword, limit=limit)
            headers = {"User-Agent": "Mozilla/5.0"}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as resp:
                    text = await resp.text()
                    soup = BeautifulSoup(text, "html.parser")
                    results = []
                    for item in soup.select(".box-result")[:limit]:
                        title_tag = item.select_one("h2 a")
                        desc_tag = item.select_one(".content")
                        if title_tag:
                            results.append({
                                "source": source_name,
                                "title": title_tag.get_text(strip=True),
                                "url": title_tag.get("href"),
                                "description": desc_tag.get_text(strip=True) if desc_tag else "",
                                "timestamp": datetime.now().isoformat()
                            })
                    return results
        except Exception as e:
            logger.error(f"Error in fetch_news ({source_name}): {str(e)}")
            return []

    tasks = [fetch_news(name, url) for name, url in NEWS_SOURCES.items()]
    all_results = await asyncio.gather(*tasks)
    
    seen_titles = set()
    unique_results = []
    
    for source_results in all_results:
        for news in source_results:
            is_similar = False
            for seen_news in unique_results:
                similarity = SequenceMatcher(None, news["title"], seen_news["title"]).ratio()
                if similarity > 0.8:
                    is_similar = True
                    break
            
            if news["title"] not in seen_titles and not is_similar:
                seen_titles.add(news["title"])
                unique_results.append(news)
                
    return unique_results[:limit]

@mcp.tool()
async def get_news_content(url: str) -> str:
    """获取新闻内容"""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        
        content = ""
        for cls in ["article", "article-content", "main-content", "content"]:
            node = soup.find(class_=cls)
            if node:
                content = node.get_text(separator="\n", strip=True)
                break
                
        if not content:
            paragraphs = soup.find_all("p")
            content = "\n".join([p.get_text() for p in paragraphs if len(p.get_text()) > 30])
            
        return content[:2000]
    except Exception as e:
        logger.error(f"Error in get_news_content: {str(e)}")
        return ""

@mcp.tool()
async def summarize_text(text: str) -> str:
    """智能新闻摘要，使用TextRank算法提取关键句子"""
    if len(text) < 200:
        return text

    sentences = text.split("。")
    keywords = jieba.analyse.textrank(text, topK=20)
    scored_sentences = []

    for sentence in sentences:
        score = sum(1 for keyword in keywords if keyword in sentence)
        scored_sentences.append((score, sentence))

    summary_sentences = sorted(scored_sentences, reverse=True)[:3]
    summary = "。".join(sentence for score, sentence in sorted(summary_sentences, key=lambda x: sentences.index(x[1]))) + "。"

    return summary

@mcp.tool()
async def analyze_sentiment(text: str) -> Dict[str, Any]:
    """增强版情感分析，返回详细的情感分析结果"""
    sentiment_dict = {
        "正面": ["优秀", "突破", "创新", "增长", "机遇", "利好", "向好", "提升", "优化", "表彰",
                "成功", "领先", "第一", "突出", "优质", "卓越", "发展", "进步", "机会", "效益"],
        "负面": ["下滑", "危机", "损失", "困境", "质疑", "下跌", "违规", "处罚", "风险", "退市",
                "失败", "亏损", "问题", "困难", "纠纷", "投诉", "起诉", "违法", "下降", "低迷"],
        "中性": ["公告", "表示", "报道", "透露", "预计", "预期", "预测", "显示", "统计", "发布",
                "介绍", "称", "表态", "回应", "强调", "指出", "提到", "认为", "分析"]
    }
    
    scores = defaultdict(int)
    words = jieba.lcut(text)
    
    for word in words:
        for sentiment, word_list in sentiment_dict.items():
            if word in word_list:
                scores[sentiment] += 1
                
    max_score = max(scores.values()) if scores else 0
    main_sentiment = "中性"
    if max_score > 0:
        main_sentiment = max(scores.items(), key=lambda x: x[1])[0]
        
    total_scores = sum(scores.values())
    intensity = max_score / total_scores if total_scores > 0 else 0
    
    return {
        "sentiment": main_sentiment,
        "intensity": round(intensity, 2),
        "details": dict(scores)
    }

@mcp.tool()
async def get_hot_topics(limit: int = 5) -> List[Dict[str, Any]]:
    """发现热门新闻话题"""
    try:
        url = "https://s.weibo.com/top/summary"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        
        topics = []
        for item in soup.select(".td-02")[:limit]:
            topic = item.get_text(strip=True)
            if topic:
                related_news = await search_news(topic, limit=3)
                topics.append({
                    "topic": topic,
                    "related_news": related_news
                })
        
        return topics
        
    except Exception as e:
        logger.error(f"Error getting hot topics: {str(e)}")
        return []

def classify_news(text: str) -> str:
    """对新闻进行分类"""
    scores = defaultdict(int)
    for category, keywords in NEWS_CATEGORIES.items():
        for keyword in keywords:
            scores[category] += text.count(keyword)
            
    return max(scores.items(), key=lambda x: x[1])[0] if scores else "其他"

if __name__ == "__main__":
    try:
        logger.info("Starting news aggregator MCP server...")
        mcp.run(transport='stdio')
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
@mcp.tool("search_news")
async def search_news(keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
    """搜索新闻接口
    
    Args:
        keyword: 搜索关键词
        limit: 返回结果数量限制
        
    Returns:
        List[Dict]: 新闻列表
    """
    # 清理过期缓存
    clean_expired_cache()

    async def fetch_news(source_name: str, url_template: str) -> List[Dict[str, Any]]:
        try:
            url = url_template.format(keyword=keyword, limit=limit)
            headers = {"User-Agent": "Mozilla/5.0"}
            # 使用aiohttp替代requests进行异步请求
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as resp:
                    text = await resp.text()
                    soup = BeautifulSoup(text, "html.parser")
                    results = []
                    for item in soup.select(".box-result")[:limit]:
                        title_tag = item.select_one("h2 a")
                        desc_tag = item.select_one(".content")
                        if title_tag:
                            results.append({
                                "source": source_name,
                                "title": title_tag.get_text(strip=True),
                                "url": title_tag.get("href"),
                                "description": desc_tag.get_text(strip=True) if desc_tag else "",
                                "timestamp": datetime.now().isoformat()
                            })
                    return results
        except Exception as e:
            logger.error(f"Error in fetch_news ({source_name}): {str(e)}")
            return []

    # 并发获取所有新闻源的结果
    tasks = [fetch_news(name, url) for name, url in NEWS_SOURCES.items()]
    all_results = await asyncio.gather(*tasks)
    
    # 合并结果并去重
    seen_titles = set()
    unique_results = []
    
    for source_results in all_results:
        for news in source_results:
            # 检查标题相似度
            is_similar = False
            for seen_news in unique_results:
                similarity = SequenceMatcher(None, news["title"], seen_news["title"]).ratio()
                if similarity > 0.8:  # 相似度阈值
                    is_similar = True
                    break
            
            if news["title"] not in seen_titles and not is_similar:
                seen_titles.add(news["title"])
                unique_results.append(news)
                
    return unique_results[:limit]

@mcp.tool()
async def get_news_content(url: str) -> str:
    """获取新闻内容"""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        
        content = ""
        for cls in ["article", "article-content", "main-content", "content"]:
            node = soup.find(class_=cls)
            if node:
                content = node.get_text(separator="\n", strip=True)
                break
                
        if not content:
            paragraphs = soup.find_all("p")
            content = "\n".join([p.get_text() for p in paragraphs if len(p.get_text()) > 30])
            
        return content[:2000]  # 限制长度
    except Exception as e:
        logger.error(f"Error in get_news_content: {str(e)}")
        return ""

@mcp.tool()
async def summarize_text(text: str) -> str:
    """智能新闻摘要，使用TextRank算法提取关键句子"""
    if len(text) < 200:
        return text
        
    sentences = text.split("。")
    # 使用TextRank算法选择最重要的句子
    keywords = jieba.analyse.textrank(text, topK=20)
    scored_sentences = []
    
    for sentence in sentences:
        score = sum(1 for keyword in keywords if keyword in sentence)
        scored_sentences.append((score, sentence))
        
    # 选择得分最高的3个句子
    summary_sentences = sorted(scored_sentences, reverse=True)[:3]
    summary = "。".join(sentence for score, sentence in sorted(summary_sentences, key=lambda x: sentences.index(x[1]))) + "。"
    
    return summary

@mcp.tool()
async def analyze_sentiment(text: str) -> Dict[str, Any]:
    """增强版情感分析，返回详细的情感分析结果"""
    # 扩展情感词典
    sentiment_dict = {
        "正面": ["优秀", "突破", "创新", "增长", "机遇", "利好", "向好", "提升", "优化", "表彰",
                "成功", "领先", "第一", "突出", "优质", "卓越", "发展", "进步", "机会", "效益"],
        "负面": ["下滑", "危机", "损失", "困境", "质疑", "下跌", "违规", "处罚", "风险", "退市",
                "失败", "亏损", "问题", "困难", "纠纷", "投诉", "起诉", "违法", "下降", "低迷"],
        "中性": ["公告", "表示", "报道", "透露", "预计", "预期", "预测", "显示", "统计", "发布",
                "介绍", "称", "表态", "回应", "强调", "指出", "提到", "认为", "分析"]
    }
    
    # 计算情感得分
    scores = defaultdict(int)
    words = jieba.lcut(text)
    
    for word in words:
        for sentiment, word_list in sentiment_dict.items():
            if word in word_list:
                scores[sentiment] += 1
                
    # 确定主要情感
    max_score = max(scores.values()) if scores else 0
    main_sentiment = "中性"
    if max_score > 0:
        main_sentiment = max(scores.items(), key=lambda x: x[1])[0]
        
    # 计算情感强度
    total_scores = sum(scores.values())
    intensity = max_score / total_scores if total_scores > 0 else 0
    
    return {
        "sentiment": main_sentiment,
        "intensity": round(intensity, 2),
        "details": dict(scores)
    }

@mcp.tool()
async def get_hot_topics(limit: int = 5) -> List[Dict[str, Any]]:
    """发现热门新闻话题"""
    try:
        # 获取热搜榜单
        url = "https://s.weibo.com/top/summary"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        
        topics = []
        for item in soup.select(".td-02")[:limit]:
            topic = item.get_text(strip=True)
            if topic:
                # 获取每个话题的相关新闻
                related_news = await search_news(topic, limit=3)
                topics.append({
                    "topic": topic,
                    "related_news": related_news
                })
        
        return topics
        
    except Exception as e:
        logger.error(f"Error getting hot topics: {str(e)}")
        return []

def classify_news(text: str) -> str:
    """对新闻进行分类"""
    scores = defaultdict(int)
    for category, keywords in NEWS_CATEGORIES.items():
        for keyword in keywords:
            scores[category] += text.count(keyword)
            
    return max(scores.items(), key=lambda x: x[1])[0] if scores else "其他"

if __name__ == "__main__":
    try:
        logger.info("Starting news aggregator MCP server...")
        mcp.run(transport='stdio')
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")

@mcp.tool("search_news")
async def search_news(keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
    """搜索新闻接口
    
    Args:
        keyword: 搜索关键词
        limit: 返回结果数量限制
        
    Returns:
        List[Dict]: 新闻列表
    """
    # 清理过期缓存
    clean_expired_cache()

    async def fetch_news(source_name: str, url_template: str) -> List[Dict[str, Any]]:
        try:
            url = url_template.format(keyword=keyword, limit=limit)
            headers = {"User-Agent": "Mozilla/5.0"}
            # 使用aiohttp替代requests进行异步请求
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as resp:
                    text = await resp.text()
                    soup = BeautifulSoup(text, "html.parser")
                    results = []
                    for item in soup.select(".box-result")[:limit]:
                        title_tag = item.select_one("h2 a")
                        desc_tag = item.select_one(".content")
                        if title_tag:
                            results.append({
                                "source": source_name,
                                "title": title_tag.get_text(strip=True),
                                "url": title_tag.get("href"),
                                "description": desc_tag.get_text(strip=True) if desc_tag else "",
                                "timestamp": datetime.now().isoformat()
                            })
                    return results
        except Exception as e:
            logger.error(f"Error in fetch_news ({source_name}): {str(e)}")
            return []

    # 并发获取所有新闻源的结果
    tasks = [fetch_news(name, url) for name, url in NEWS_SOURCES.items()]
    all_results = await asyncio.gather(*tasks)
    
    # 合并结果并去重
    seen_titles = set()
    unique_results = []
    
    for source_results in all_results:
        for news in source_results:
            # 检查标题相似度
            is_similar = False
            for seen_news in unique_results:
                similarity = SequenceMatcher(None, news["title"], seen_news["title"]).ratio()
                if similarity > 0.8:  # 相似度阈值
                    is_similar = True
                    break
            
            if news["title"] not in seen_titles and not is_similar:
                seen_titles.add(news["title"])
                unique_results.append(news)
                
    return unique_results[:limit]

@mcp.tool()
async def get_news_content(url: str) -> str:
    """
    获取新闻内容
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        
        content = ""
        for cls in ["article", "article-content", "main-content", "content"]:
            node = soup.find(class_=cls)
            if node:
                content = node.get_text(separator="\n", strip=True)
                break
                
        if not content:
            paragraphs = soup.find_all("p")
            content = "\n".join([p.get_text() for p in paragraphs if len(p.get_text()) > 30])
            
        return content[:2000]  # 限制长度
    except Exception as e:
        logger.error(f"Error in get_news_content: {str(e)}")
        return ""

@mcp.tool("search_news")
async def search_news(keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
    """搜索新闻接口
    
    Args:
        keyword: 搜索关键词
        limit: 返回结果数量限制
        
    Returns:
        List[Dict]: 新闻列表
    """
    # 清理过期缓存
    clean_expired_cache()

    async def fetch_news(source_name: str, url_template: str) -> List[Dict[str, Any]]:
        try:
            url = url_template.format(keyword=keyword, limit=limit)
            headers = {"User-Agent": "Mozilla/5.0"}
            # 使用aiohttp替代requests进行异步请求
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as resp:
                    text = await resp.text()
                    soup = BeautifulSoup(text, "html.parser")
                    results = []
                    for item in soup.select(".box-result")[:limit]:
                        title_tag = item.select_one("h2 a")
                        desc_tag = item.select_one(".content")
                        if title_tag:
                            results.append({
                                "source": source_name,
                                "title": title_tag.get_text(strip=True),
                                "url": title_tag.get("href"),
                                "description": desc_tag.get_text(strip=True) if desc_tag else "",
                                "timestamp": datetime.now().isoformat()
                            })
                    return results
        except Exception as e:
            logger.error(f"Error in fetch_news ({source_name}): {str(e)}")
            return []

    # 并发获取所有新闻源的结果
    tasks = [fetch_news(name, url) for name, url in NEWS_SOURCES.items()]
    all_results = await asyncio.gather(*tasks)
    
    # 合并结果并去重
    seen_titles = set()
    unique_results = []
    
    for source_results in all_results:
        for news in source_results:
            # 检查标题相似度
            is_similar = False
            for seen_news in unique_results:
                similarity = SequenceMatcher(None, news["title"], seen_news["title"]).ratio()
                if similarity > 0.8:  # 相似度阈值
                    is_similar = True
                    break
            
            if news["title"] not in seen_titles and not is_similar:
                seen_titles.add(news["title"])
                unique_results.append(news)
                
    return unique_results[:limit]


@mcp.tool()
async def summarize_text(text: str) -> str:
    """
    智能新闻摘要，使用TextRank算法提取关键句子
    """
    if len(text) < 200:
        return text
        
    sentences = text.split("。")
    # 使用TextRank算法选择最重要的句子
    keywords = jieba.analyse.textrank(text, topK=20)
    scored_sentences = []
    
    for sentence in sentences:
        score = sum(1 for keyword in keywords if keyword in sentence)
        scored_sentences.append((score, sentence))
        
    # 选择得分最高的3个句子
    summary_sentences = sorted(scored_sentences, reverse=True)[:3]
    summary = "。".join(sentence for score, sentence in sorted(summary_sentences, key=lambda x: sentences.index(x[1]))) + "。"
    
    return summary

@mcp.tool()
async def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    增强版情感分析，返回详细的情感分析结果
    """
    # 扩展情感词典
    sentiment_dict = {
        "正面": ["优秀", "突破", "创新", "增长", "机遇", "利好", "向好", "提升", "优化", "表彰",
                "成功", "领先", "第一", "突出", "优质", "卓越", "发展", "进步", "机会", "效益"],
        "负面": ["下滑", "危机", "损失", "困境", "质疑", "下跌", "违规", "处罚", "风险", "退市",
                "失败", "亏损", "问题", "困难", "纠纷", "投诉", "起诉", "违法", "下降", "低迷"],
        "中性": ["公告", "表示", "报道", "透露", "预计", "预期", "预测", "显示", "统计", "发布",
                "介绍", "称", "表态", "回应", "强调", "指出", "提到", "认为", "分析"]
    }
    
    # 计算情感得分
    scores = defaultdict(int)
    words = jieba.lcut(text)
    
    for word in words:
        for sentiment, word_list in sentiment_dict.items():
            if word in word_list:
                scores[sentiment] += 1
                
    # 确定主要情感
    max_score = max(scores.values()) if scores else 0
    main_sentiment = "中性"
    if max_score > 0:
        main_sentiment = max(scores.items(), key=lambda x: x[1])[0]
        
    # 计算情感强度
    total_scores = sum(scores.values())
    intensity = max_score / total_scores if total_scores > 0 else 0
    
    return {
        "sentiment": main_sentiment,
        "intensity": round(intensity, 2),
        "details": dict(scores)
    }

def get_cached_data(key: str) -> Optional[Dict]:
    """从缓存获取数据"""
    cache_file = os.path.join(CACHE_DIR, f"{hashlib.md5(key.encode()).hexdigest()}.json")
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if time.time() - data['timestamp'] < CACHE_EXPIRE:
                return data['content']
    return None

def save_to_cache(key: str, content: Any):
    """保存数据到缓存"""
    cache_file = os.path.join(CACHE_DIR, f"{hashlib.md5(key.encode()).hexdigest()}.json")
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': time.time(),
            'content': content
        }, f, ensure_ascii=False)

def clean_expired_cache():
    """清理过期的缓存"""
    try:
        current_time = time.time()
        expired_keys = [k for k, v in news_cache.items() if current_time - v['timestamp'] > CACHE_EXPIRY]
        for key in expired_keys:
            del news_cache[key]
        if expired_keys:
            logger.debug(f"Cleaned {len(expired_keys)} expired cache entries")
    except Exception as e:
        logger.error(f"Cache cleanup error: {str(e)}")
@mcp.tool()
async def get_hot_topics(limit: int = 5) -> List[Dict[str, Any]]:
    """
    发现热门新闻话题
    """
    try:
        # 获取热搜榜单
        url = "https://s.weibo.com/top/summary"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        
        topics = []
        for item in soup.select(".td-02")[:limit]:
            topic = item.get_text(strip=True)
            if topic:
                # 获取每个话题的相关新闻
                related_news = await search_news(topic, limit=3)
                topics.append({
                    "topic": topic,
                    "related_news": related_news
                })
        
        return topics
        
    except Exception as e:
        logger.error(f"Error getting hot topics: {str(e)}")
        return []

def classify_news(text: str) -> str:
    """
    对新闻进行分类
    """
    scores = defaultdict(int)
    for category, keywords in NEWS_CATEGORIES.items():
        for keyword in keywords:
            scores[category] += text.count(keyword)
            
    return max(scores.items(), key=lambda x: x[1])[0] if scores else "其他"

def clean_expired_cache():
    """清理过期的缓存文件"""
    try:
        current_time = time.time()
        for filename in os.listdir(CACHE_DIR):
            file_path = os.path.join(CACHE_DIR, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if current_time - data['timestamp'] > CACHE_EXPIRE:
                        os.remove(file_path)
                        logger.info(f"Removed expired cache file: {filename}")
    except Exception as e:
        logger.error(f"Error in clean_expired_cache: {str(e)}")

def start_cache_cleanup():
    """启动缓存清理任务"""
    async def cleanup_task():
        while True:
            try:
                # 清理过期的缓存
                current_time = time.time()
                expired_keys = [k for k, v in news_cache.items() if current_time - v['timestamp'] > CACHE_EXPIRY]
                for key in expired_keys:
                    del news_cache[key]
                logger.debug(f"Cleaned {len(expired_keys)} expired cache entries")
                # 等待下一次清理
                await asyncio.sleep(CLEANUP_INTERVAL)
            except Exception as e:
                logger.error(f"Cache cleanup error: {str(e)}")
                await asyncio.sleep(60)  # 发生错误时等待较长时间

    # 在后台启动清理任务
    loop = asyncio.get_event_loop()
    loop.create_task(cleanup_task())
start_cache_cleanup()

@mcp.tool()
async def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    增强版情感分析，返回详细的情感分析结果
    """
    # 情感词典
    positive_words = set(["好", "优秀", "出色", "成功", "突破", "创新", "发展", "提升", "增长", "利好"])
    negative_words = set(["差", "糟糕", "失败", "下跌", "损失", "风险", "问题", "困难", "挑战", "担忧"])
    
    # 分词
    words = jieba.lcut(text)
    
    # 统计情感词出现次数
    pos_count = sum(1 for word in words if word in positive_words)
    neg_count = sum(1 for word in words if word in negative_words)
    
    # 计算情感得分 (-1到1之间)
    total = pos_count + neg_count
    if total == 0:
        sentiment_score = 0
    else:
        sentiment_score = (pos_count - neg_count) / total
        
    # 确定情感标签
    if sentiment_score > 0.2:
        sentiment = "积极"
    elif sentiment_score < -0.2:
        sentiment = "消极"
    else:
        sentiment = "中性"
        
    return {
        "sentiment": sentiment,
        "score": sentiment_score,
        "positive_count": pos_count,
        "negative_count": neg_count,
        "details": {
            "positive_words": [w for w in words if w in positive_words],
            "negative_words": [w for w in words if w in negative_words]
        }
    }

@mcp.tool()
async def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    增强版情感分析，返回详细的情感分析结果
    """
    # 情感词典
    positive_words = set(["好", "优秀", "出色", "成功", "突破", "创新", "发展", "提升", "增长", "利好"])
    negative_words = set(["差", "糟糕", "失败", "下跌", "损失", "风险", "问题", "困难", "挑战", "担忧"])
    
    # 分词
    words = jieba.lcut(text)
    
    # 统计情感词出现次数
    pos_count = sum(1 for word in words if word in positive_words)
    neg_count = sum(1 for word in words if word in negative_words)
    
    # 计算情感得分 (-1到1之间)
    total = pos_count + neg_count
    if total == 0:
        sentiment_score = 0
    else:
        sentiment_score = (pos_count - neg_count) / total
        
    # 确定情感标签
    if sentiment_score > 0.2:
        sentiment = "积极"
    elif sentiment_score < -0.2:
        sentiment = "消极"
    else:
        sentiment = "中性"
        
    return {
        "sentiment": sentiment,
        "score": sentiment_score,
        "positive_count": pos_count,
        "negative_count": neg_count,
        "details": {
            "positive_words": [w for w in words if w in positive_words],
            "negative_words": [w for w in words if w in negative_words]
        }
    }

if __name__ == "__main__":
    try:
        logger.info("Starting news aggregator MCP server...")
        # 直接运行MCP服务器
        mcp.run(transport='stdio')
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
