import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob
from sklearn.metrics import accuracy_score, precision_score, recall_score
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# 1. 결정트리 함수
def calcEntropy(classLabels):
    length = len(classLabels)
    if length == 0:
        return 0
    classCounts = classLabels.value_counts().to_dict()
    entropy = -sum((count / length) * math.log2(count / length) for count in classCounts.values())
    return entropy

def calcInformationGain(data, feature, classLabel):
    infoD = calcEntropy(data[classLabel])
    featureValues = data[feature].unique()
    infoDj = sum(len(data[data[feature] == val]) / len(data) * calcEntropy(data[data[feature] == val][classLabel]) for val in featureValues)
    return infoD - infoDj

def getBestFeature(data, features, classLabel):
    return max(features, key=lambda f: calcInformationGain(data, f, classLabel) if f != classLabel else -1)

def buildDecisionTree(data, features, classLabel):
    if len(data[classLabel].unique()) == 1:
        return data[classLabel].iloc[0]
    if not features:
        return data[classLabel].mode()[0]
    feature = getBestFeature(data, features, classLabel)
    tree = {feature: {}}
    for value in data[feature].unique():
        subset = data[data[feature] == value]
        subFeatures = [f for f in features if f != feature]
        tree[feature][value] = buildDecisionTree(subset, subFeatures, classLabel)
    return tree

def getTestResult(tree, rowData):
    if not isinstance(tree, dict):
        return tree
    root = list(tree.keys())[0]
    value = rowData[root]
    return getTestResult(tree[root][value], rowData) if value in tree[root] else None

# 2. 데이터 전처리 (청크 단위 처리)
def preprocess_data_chunk(chunk):
    chunk['sentiment'] = chunk['text'].apply(lambda x: 'positive' if TextBlob(str(x)).sentiment.polarity > 0 else 'negative')
    chunk['text_length'] = chunk['text'].apply(len)
    chunk['label'] = chunk.apply(lambda row: 'inquiry' if row['inbound'] else 'response', axis=1)
    return chunk[['text', 'text_length', 'sentiment', 'label', 'author_id']]

# 3. BERT 임베딩 (배치 처리)
class BERTEmbedding:
    def __init__(self, batch_size=16):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def get_embeddings(self, texts):
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings.extend(outputs.last_hidden_state[:, 0, :].cpu().numpy())
        return np.array(embeddings)

# 4. LSTM 모델
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class TextDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

def train_lstm_model(embeddings, labels, num_classes, batch_size=32):
    model = LSTMClassifier(embeddings.shape[1], 128, num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    dataset = TextDataset(embeddings, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(5):
        logging.info(f"Epoch {epoch+1}/5")
        for batch_embeddings, batch_labels in dataloader:
            batch_embeddings = batch_embeddings.unsqueeze(1).to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_embeddings)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
    return model

# 5. 응답 함수
def suggest_response(predicted_label, text, author_id):
    if predicted_label != 'inquiry':
        return "Thank you for your feedback! We're here to help if you need anything else."

    # author_id에 따라 문의 유형 세분화
    text_lower = text.lower()
    if 'AppleSupport' in author_id:
        if 'ios' in text_lower or 'update' in text_lower:
            return "We’re sorry for the inconvenience with your iOS update. Please check our support page for troubleshooting: [iOS Support Link]."
        elif 'battery' in text_lower:
            return "Battery issues can be frustrating. Let’s troubleshoot—please DM us your device details: [DM Link]."
        else:
            return "We’re here to assist with your Apple device. Could you provide more details? [DM Link]"

    elif 'Uber_Support' in author_id:
        if 'ride' in text_lower or 'driver' in text_lower:
            return "We apologize for the issue with your ride. Please share more details via DM so we can assist: [DM Link]."
        elif 'payment' in text_lower:
            return "It seems there’s an issue with your payment. Please contact our support team: [Uber Payment Support]."
        else:
            return "We’re here to help with your Uber experience. Please provide more details: [DM Link]."

    elif 'Delta' in author_id:
        if 'flight' in text_lower or 'delay' in text_lower:
            return "We’re sorry for the delay with your flight. Check the latest updates here: [Flight Status Link]."
        elif 'booking' in text_lower:
            return "Having trouble with your booking? Let’s resolve it—please DM us: [DM Link]."
        else:
            return "We’re here to assist with your Delta flight. Please share more details: [DM Link]."

    elif 'SpotifyCares' in author_id:
        if 'playback' in text_lower or 'skipping' in text_lower:
            return "Sorry for the playback issues. Try logging out, restarting your device, and logging back in. Let us know if it persists!"
        else:
            return "We’re here to help with your Spotify experience. Could you share more details? [DM Link]"

    elif 'VirginTrains' in author_id:
        if 'error' in text_lower or 'website' in text_lower:
            return "We apologize for the website error. Try using this link to book: [Booking Link], or contact us at ~~."
        else:
            return "We’re here to assist with your Virgin Trains journey. Please provide more details: [DM Link]."

    # 일반적인 키워드 기반 응답
    if 'error' in text_lower:
        return "We apologize for the issue. Please check our FAQ: [FAQ Link]."
    elif 'delivery' in text_lower:
        return "You can track your delivery here: [Tracking Link]."
    elif 'refund' in text_lower:
        return "We’re sorry for the inconvenience. Here’s how you can request a refund: [Refund Policy Link]."
    elif 'payment' in text_lower:
        return "It seems there’s an issue with your payment. Please contact our billing team at cool.com."
    else:
        return "Thank you for reaching out. Could you provide more details so we can assist you better?"

# 6. 모델 평가 및 결과 저장
def evaluate_and_save_results(model, embeddings, labels, label_encoder, texts, author_ids, output_file='prediction_results.csv', eval_file='evaluation_results.txt'):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        test_embeddings = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(1).to(device)
        outputs = model(test_embeddings)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()

    # 원래 레이블 복원
    decoded_predictions = label_encoder.inverse_transform(predictions)
    decoded_labels = label_encoder.inverse_transform(labels)

    # 성능 평가
    accuracy = accuracy_score(decoded_labels, decoded_predictions)
    precision = precision_score(decoded_labels, decoded_predictions, average='weighted')
    recall = recall_score(decoded_labels, decoded_predictions, average='weighted')

    # 평가 결과 출력 및 파일 저장
    eval_results = (
        f"Evaluation Results:\n"
        f"Accuracy: {accuracy:.2f}\n"
        f"Precision: {precision:.2f}\n"
        f"Recall: {recall:.2f}\n"
    )
    print(eval_results)
    with open(eval_file, 'w', encoding='utf-8') as f:
        f.write(eval_results)

    # 모든 샘플에 대해 응답 생성 및 저장
    results = []
    for text, pred, author_id in zip(texts, decoded_predictions, author_ids):
        response = suggest_response(pred, text, author_id)
        results.append({
            'text': text,
            'predicted_label': pred,
            'response': response,
            'author_id': author_id
        })

    # 결과를 DataFrame으로 변환 및 CSV 저장
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False, encoding='utf-8')
    logging.info(f"Prediction results saved to {output_file}")

# 메인 함수
def main(file_path, sample_size=5000):
    logging.info("Starting data processing...")

    # 데이터 전처리
    chunksize = 5000
    processed_data = []
    total_rows = 0
    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        processed_chunk = preprocess_data_chunk(chunk)
        processed_data.append(processed_chunk.sample(min(sample_size - total_rows, len(processed_chunk))))
        total_rows += len(processed_chunk)
        if total_rows >= sample_size:
            break

    data = pd.concat(processed_data)
    logging.info(f"Processed {len(data)} rows from {total_rows} total rows")

    texts = data['text'].tolist()
    labels = data['label']
    author_ids = data['author_id'].tolist()
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # 결정트리
    features = ['text_length', 'sentiment']
    tree = buildDecisionTree(data, features, 'label')
    logging.info("Decision Tree built")

    # BERT 임베딩
    bert = BERTEmbedding(batch_size=16)
    embeddings = bert.get_embeddings(texts)
    logging.info("BERT embeddings generated")

    # LSTM 모델 학습
    lstm_model = train_lstm_model(embeddings, encoded_labels, len(label_encoder.classes_), batch_size=32)
    logging.info("LSTM model trained")

    # 모델 평가 및 결과 저장
    evaluate_and_save_results(
        lstm_model, embeddings, encoded_labels, label_encoder, texts, author_ids,
        output_file='prediction_results.csv',
        eval_file='evaluation_results.txt'
    )

if __name__ == "__main__":
    main('./dataset/twcs/customer_support_twitter.csv')