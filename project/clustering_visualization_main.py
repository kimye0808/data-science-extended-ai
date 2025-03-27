import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
import umap
from sklearn.metrics import silhouette_score

# 경고 억제
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "2"

# 데이터 로드
df = pd.read_csv("./dataset/twcs/customer_support_twitter.csv")

# 고객 문의만 필터링 (inbound == True)
df_inbound = df[df['inbound'] == True]

# 1,000개 샘플링
sample_size = 1000
df_sample = df_inbound.sample(n=sample_size, random_state=42)

# 텍스트 데이터 추출
texts = df_sample['text'].tolist()

# 영어 텍스트 전처리 개선
def preprocess_text(texts):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    processed_texts = []
    for text in texts:
        text = str(text).lower()
        # URL, @, # 제거
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        # 특수문자 제거 및 토큰화
        text = ''.join(c for c in text if c.isalnum() or c.isspace())
        text = ' '.join(lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words)
        processed_texts.append(text)
    return processed_texts

texts = preprocess_text(texts)

# BERT로 텍스트 임베딩
class BERTEmbedding:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def get_embeddings(self, texts, batch_size=32):
        embeddings = []
        full_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                for j in range(len(batch_texts)):
                    embeddings.append((batch_texts[j], batch_embeddings[j]))
                    full_embeddings.append(batch_embeddings[j])
        return embeddings, np.array(full_embeddings)

# 임베딩 생성
bert = BERTEmbedding()
embeddings, full_embeddings = bert.get_embeddings(texts)

# UMAP으로 차원 축소 ( -> 10차원)
reducer = umap.UMAP(n_components=10, metric='cosine', random_state=42)
reduced_embeddings = reducer.fit_transform(full_embeddings)

# DBSCAN
def eucDistanceFunc(a, b):
    a_vec = a[1]  # 차원 축소된 10차원 벡터
    b_vec = b[1]
    return 1 - cosine_similarity([a_vec], [b_vec])[0][0]

def rangeQuery(db, p, eps):
    neighborsN = []
    for i in range(len(db)):
        if eucDistanceFunc(db[p], db[i]) <= eps:
            neighborsN.append(i)
    return neighborsN

def dbscan(db, eps, minPts):
    labels = len(db) * [None]
    c = 0

    for p in tqdm(range(len(db)), desc="DBSCAN"):
        if labels[p] is not None:
            continue

        neighborsN = rangeQuery(db, p, eps)
        if len(neighborsN) < minPts:
            labels[p] = -1
            continue

        c += 1
        labels[p] = c

        seedSet = set(neighborsN) - {p}
        while seedSet:
            q = seedSet.pop()
            if labels[q] == -1:
                labels[q] = c
            if labels[q] is not None:
                continue

            labels[q] = c
            neighborsQ = rangeQuery(db, q, eps)
            if len(neighborsQ) < minPts:
                continue
            seedSet.update(set(neighborsQ))

    return labels

# 차원 축소된 임베딩으로 DBSCAN 입력 데이터 준비
reduced_embeddings_list = [(text, emb) for text, emb in zip(texts, reduced_embeddings)]

# DBSCAN 실행
eps = 0.002 
minPts = 5 
labels = dbscan(reduced_embeddings_list, eps, minPts)

# 클러스터링 결과 분석
unique_labels = set(labels)
print(f"Number of clusters: {len(unique_labels) - (1 if -1 in unique_labels else 0)}")
print(f"Number of noise points: {list(labels).count(-1)}")

# 실루엣 스코어 계산
# Noise(-1)로 분류된 데이터는 제외
non_noise_indices = [i for i, label in enumerate(labels) if label != -1]
non_noise_labels = [labels[i] for i in non_noise_indices]
non_noise_embeddings = reduced_embeddings[non_noise_indices]

if len(set(non_noise_labels)) > 1:  # 클러스터가 2개 이상이어야 실루엣 스코어 계산 가능
    silhouette_avg = silhouette_score(non_noise_embeddings, non_noise_labels, metric='cosine')
    print(f"Silhouette Score (excluding noise): {silhouette_avg:.3f}")
else:
    print("Silhouette Score cannot be calculated: Not enough clusters (need at least 2 clusters excluding noise).")

# 클러스터별 유형 식별
def identify_cluster_types(texts, labels):
    cluster_texts = {}
    for i, label in enumerate(labels):
        if label != -1:
            if label not in cluster_texts:
                cluster_texts[label] = []
            cluster_texts[label].append(texts[i])

    cluster_types = {}
    for cid, cluster_text in cluster_texts.items():
        combined_text = " ".join(cluster_text).lower()
        # 키워드 조합으로 더 구체적으로 유형 식별 
        if any(kw in combined_text for kw in ["order", "delivery", "package", "shipping", "arrive", "delay", "shipment", "tracking number"]):
            if any(kw in combined_text for kw in ["track", "tracking", "where", "status"]):
                cluster_types[cid] = "Order Tracking"
            elif any(kw in combined_text for kw in ["delay", "late", "arrive"]):
                cluster_types[cid] = "Delivery Delay"
            else:
                cluster_types[cid] = "Delivery Inquiry"
        elif any(kw in combined_text for kw in ["payment", "bill", "charge", "invoice", "transaction", "pay", "card", "declined"]):
            if any(kw in combined_text for kw in ["status", "update"]):
                cluster_types[cid] = "Payment Status"
            elif any(kw in combined_text for kw in ["declined", "failed", "error"]):
                cluster_types[cid] = "Payment Failure"
            else:
                cluster_types[cid] = "Payment Inquiry"
        elif any(kw in combined_text for kw in ["refund", "return", "cancel", "money", "back", "reimburse"]):
            if any(kw in combined_text for kw in ["complete", "done", "processed"]):
                cluster_types[cid] = "Refund Completed"
            elif any(kw in combined_text for kw in ["delay", "late", "waiting"]):
                cluster_types[cid] = "Refund Delayed"
            elif any(kw in combined_text for kw in ["denied", "reject", "not eligible"]):
                cluster_types[cid] = "Refund Denied"
            elif any(kw in combined_text for kw in ["status", "where", "update"]):
                cluster_types[cid] = "Refund Processing"
            else:
                cluster_types[cid] = "Refund Request"
        elif any(kw in combined_text for kw in ["login", "password", "account", "access", "sign", "locked", "verify"]):
            if any(kw in combined_text for kw in ["login", "sign", "access"]):
                cluster_types[cid] = "Login Issue"
            elif any(kw in combined_text for kw in ["locked"]):
                cluster_types[cid] = "Account Locked"
            elif any(kw in combined_text for kw in ["verify"]):
                cluster_types[cid] = "Verification Issue"
            else:
                cluster_types[cid] = "Account Inquiry"
        elif any(kw in combined_text for kw in ["app", "ios", "android", "update", "error", "bug", "crash", "slow", "freeze", "lag", "glitch", "down", "broken"]):
            if any(kw in combined_text for kw in ["app", "ios", "android"]):
                if any(kw in combined_text for kw in ["error", "bug", "crash", "slow", "freeze", "lag", "glitch", "down", "broken"]):
                    cluster_types[cid] = "App Issue"
                else:
                    cluster_types[cid] = "App Inquiry"
            elif any(kw in combined_text for kw in ["update"]):
                cluster_types[cid] = "Update Issue"
            else:
                cluster_types[cid] = "Device Issue"
        elif any(kw in combined_text for kw in ["service", "support", "help", "customer", "response", "call", "contact", "wait"]):
            if any(kw in combined_text for kw in ["wait", "response"]):
                cluster_types[cid] = "Support Delay"
            else:
                cluster_types[cid] = "Service Inquiry"
        elif any(kw in combined_text for kw in ["promotion", "discount", "offer", "deal", "sale"]):
            cluster_types[cid] = "Promotion Inquiry"
        elif any(kw in combined_text for kw in ["product", "item", "feature", "details", "info", "price", "availability", "stock"]):
            if any(kw in combined_text for kw in ["price"]):
                cluster_types[cid] = "Price Inquiry"
            elif any(kw in combined_text for kw in ["availability", "stock"]):
                cluster_types[cid] = "Stock Inquiry"
            else:
                cluster_types[cid] = "Product Inquiry"
        elif any(kw in combined_text for kw in ["store", "location", "address", "open", "close"]):
            if any(kw in combined_text for kw in ["where", "location", "address"]):
                cluster_types[cid] = "Store Location Inquiry"
            else:
                cluster_types[cid] = "Store Hours Inquiry"
        else:
            cluster_types[cid] = "Other"

    # 클러스터별 대표 텍스트 출력
    for cid, cluster_text in cluster_texts.items():
        print(f"Cluster {cid} ({cluster_types[cid]}): {len(cluster_text)} texts")
        print(f"Sample texts: {cluster_text[:3]}")
    return cluster_types

cluster_types = identify_cluster_types(texts, labels)

# 클러스터 유형 분포 출력
cluster_type_counts = {ctype: 0 for ctype in [
    "Order Tracking", "Delivery Delay", "Delivery Inquiry", 
    "Payment Status", "Payment Failure", "Payment Inquiry", 
    "Service Inquiry", "Support Delay", 
    "App Issue", "App Inquiry", "Update Issue", "Device Issue", 
    "Login Issue", "Account Locked", "Verification Issue", "Account Inquiry", 
    "Refund Processing", "Refund Delayed", "Refund Completed", "Refund Denied", "Refund Request", 
    "Price Inquiry", "Stock Inquiry", "Product Inquiry", 
    "Promotion Inquiry", 
    "Store Location Inquiry", "Store Hours Inquiry", 
    "Other", "Noise"
]}
for label in labels:
    if label == -1:
        cluster_type_counts["Noise"] += 1
    else:
        ctype = cluster_types.get(label, "Other")
        cluster_type_counts[ctype] += 1
print("Cluster type distribution:", cluster_type_counts)

# 규칙 기반 응답 생성 (세분화)
def generate_response(inquiry_type):
    responses = {
        "Order Tracking": "Please track your order here: [Tracking Link].",
        "Delivery Delay": "We’re sorry for the delay. Please check your delivery status: [Tracking Link].",
        "Delivery Inquiry": "Let us help with your delivery inquiry. Please provide more details: [Support Link].",
        "Payment Status": "You can check your payment status here: [Payment Status Link].",
        "Payment Failure": "It seems your payment failed. Please try again or contact support: [Support Link].",
        "Payment Inquiry": "Let us help with your payment inquiry. Please contact our billing team: [Billing Link].",
        "Service Inquiry": "We’re here to help with your service inquiry. Contact us: [Support Link].",
        "Support Delay": "We’re sorry for the delay in support. We’ll respond to you shortly.",
        "App Issue": "We’re sorry for the app issue. Please try restarting the app or contact support: [Tech Support Link].",
        "App Inquiry": "Let us help with your app inquiry. Please provide more details: [Support Link].",
        "Update Issue": "We’re sorry for the update issue. Please try reinstalling the app or contact support: [Tech Support Link].",
        "Device Issue": "We’re sorry for the device issue. Please contact support for assistance: [Tech Support Link].",
        "Login Issue": "It seems there’s an issue with your login. Please try resetting your password: [Password Reset Link].",
        "Account Locked": "Your account appears to be locked. Please contact support to unlock it: [Support Link].",
        "Verification Issue": "It seems there’s an issue with your account verification. Please check your email for a verification link.",
        "Account Inquiry": "Let us help with your account inquiry. Please contact support: [Account Help Link].",
        "Refund Processing": "Your refund is being processed. Check the status here: [Refund Status Link].",
        "Refund Delayed": "We’re sorry for the delay in your refund. Please check the status: [Refund Status Link].",
        "Refund Completed": "Your refund has been completed. Check your account for details.",
        "Refund Denied": "We’re sorry, but your refund request was denied. Contact support for more details: [Support Link].",
        "Refund Request": "We’re sorry for the inconvenience. Here’s how to request a refund: [Refund Policy Link].",
        "Price Inquiry": "Here are the pricing details for the product: [Product Info Link].",
        "Stock Inquiry": "Let us check the stock availability for you: [Stock Check Link].",
        "Product Inquiry": "Thank you for your interest! Here are more details about the product: [Product Info Link].",
        "Promotion Inquiry": "Here are our current promotions: [Promotion Link].",
        "Store Location Inquiry": "You can find our store locations here: [Store Locator Link].",
        "Store Hours Inquiry": "Here’s more information about our store hours: [Store Info Link].",
        "Other": "Thank you for reaching out. We’ll look into your inquiry and get back to you soon.",
        "Noise": "We’ll look into your inquiry and get back to you soon."
    }
    return responses.get(inquiry_type, "We’ll look into your inquiry and get back to you soon.")

# 응답 생성
responses = [generate_response(cluster_types.get(label, "Noise") if label != -1 else "Noise") for label in labels]

# 데이터프레임 생성
plot_data = pd.DataFrame({
    'text': texts,
    'cluster': [str(label) for label in labels],
    'type': [cluster_types.get(label, "Noise") if label != -1 else "Noise" for label in labels],
    'response': responses
})

# 시각화
# TSNE로 2차원 축소
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(full_embeddings)

plot_data['x'] = embeddings_2d[:, 0]
plot_data['y'] = embeddings_2d[:, 1]

# 색상 맵 정의
color_discrete_map = {
    'Noise': 'gray',
    'Order Tracking': 'blue',
    'Delivery Delay': 'lightblue',
    'Delivery Inquiry': 'skyblue',
    'Payment Status': 'red',
    'Payment Failure': 'darkred',
    'Payment Inquiry': 'indianred',
    'Service Inquiry': 'green',
    'Support Delay': 'lightgreen',
    'App Issue': 'orange',
    'App Inquiry': 'lightsalmon',
    'Update Issue': 'darkorange',
    'Device Issue': 'gold',
    'Login Issue': 'purple',
    'Account Locked': 'darkviolet',
    'Verification Issue': 'plum',
    'Account Inquiry': 'violet',
    'Refund Processing': 'pink',
    'Refund Delayed': 'lightpink',
    'Refund Completed': 'magenta',
    'Refund Denied': 'darkmagenta',
    'Refund Request': 'hotpink',
    'Price Inquiry': 'cyan',
    'Stock Inquiry': 'lightcyan',
    'Product Inquiry': 'teal',
    'Promotion Inquiry': 'yellow',
    'Store Location Inquiry': 'brown',
    'Store Hours Inquiry': 'tan',
    'Other': 'black'
}

# Plotly Express로 2D 산점도 생성
fig_2d = px.scatter(
    plot_data,
    x='x',
    y='y',
    color='type',
    hover_data=['text', 'type', 'response'],
    title="Interactive DBSCAN Clustering of Customer Inquiries (2D)",
    labels={'x': 'TSNE Component 1', 'y': 'TSNE Component 2'},
    color_discrete_map=color_discrete_map
)

# 레이아웃 조정
fig_2d.update_layout(
    width=800,
    height=600,
    showlegend=True,
    title_x=0.5,
    legend_title_text='Cluster Type'
)

# 2D 그래프 표시
fig_2d.show()

# 2D HTML 파일로 저장
fig_2d.write_html("clustering_result_main_2d.html")

# 3D 시각화
tsne_3d = TSNE(n_components=3, random_state=42)
embeddings_3d = tsne_3d.fit_transform(full_embeddings)
plot_data['z'] = embeddings_3d[:, 2]

# Plotly Express로 3D 산점도 생성
fig_3d = px.scatter_3d(
    plot_data,
    x='x',
    y='y',
    z='z',
    color='type',
    hover_data=['text', 'type', 'response'],
    title="Interactive DBSCAN Clustering of Customer Inquiries (3D)",
    labels={'x': 'TSNE Component 1', 'y': 'TSNE Component 2', 'z': 'TSNE Component 3'},
    color_discrete_map=color_discrete_map
)

# 레이아웃 조정
fig_3d.update_layout(
    width=800,
    height=600,
    showlegend=True,
    title_x=0.5,
    legend_title_text='Cluster Type'
)

# 3D 그래프 표시
fig_3d.show()

# 3D HTML 파일로 저장
fig_3d.write_html("clustering_result_main_3d.html")