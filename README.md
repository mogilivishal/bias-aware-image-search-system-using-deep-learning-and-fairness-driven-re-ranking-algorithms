# 🔍 Bias-Aware Image Search System using Deep Learning and Fairness-Driven Re-Ranking Algorithms

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Tech Stack](https://img.shields.io/badge/Tech%20Stack-React%2C%20Flask%2C%20ResNet10-brightgreen)

## 📌 Overview

This project addresses the issue of **gender bias in image search engines** by developing a bias-aware image ranking system. Our solution leverages a **modified ResNet-10 deep learning model** to classify gender in image results and reorders these results using **three fairness-driven re-ranking algorithms**:

- **Epsilon-Greedy Algorithm**
- **Relevance-Aware Swapping Algorithm**
- **Fairness-Greedy Algorithm**

The goal is to promote equitable gender representation in search results for professional occupations, such as "CEO" or "doctor", across search engines like Google, Baidu, Naver, and Yandex.

---

## 🎯 Project Objectives

- Investigate gender bias in image search engines and its societal impact.
- Use AI to detect and quantify bias in professional image results.
- Implement re-ranking algorithms that mitigate gender bias while maintaining relevance.
- Build a fully functional web interface demonstrating real-time bias correction.

---

## 🧠 Methodology

### 🔍 Gender Detection

We utilized a **ResNet-10 based convolutional neural network** (`res10_300x300_ssd_iter_140000_fp16.caffemodel`) pre-trained on facial detection tasks, modified to classify gender in search result images.

### ⚖️ Re-ranking Algorithms

1. **Epsilon-Greedy**: Introduces randomness to disrupt gender clustering in the results.
2. **Relevance-Aware Swapping**: Considers image relevance before making swaps to ensure minimal disruption.
3. **Fairness-Greedy**: Aligns image rankings with real-world gender distributions for a given occupation using U.S. census data.

Each algorithm is iteratively applied to optimize the fairness of the top N results while minimizing changes to their perceived relevance.

---

## 💻 System Architecture

- **Frontend**: ReactJS for a responsive search interface.
- **Backend**: Flask API to serve model predictions and reranked image lists.
- **Model**: ResNet-10 SSD for face/gender detection.
- **Image Search**: Google, Baidu, Naver, and Yandex image APIs.

### 🔄 Data Flow

1. User enters a professional search query (e.g., “CEO”).
2. Images are retrieved from various search engines.
3. Gender is classified using the deep learning model.
4. Reranking algorithms are applied to promote fairness.
5. Balanced results are displayed on the front-end in real-time.

---

## 🖥️ Tech Stack

| Layer        | Technologies                        |
|--------------|-------------------------------------|
| Frontend     | ReactJS, Fetch API                  |
| Backend      | Flask, Python                       |
| Deep Learning| PyTorch, OpenCV, ResNet-10 (Caffe)  |
| APIs         | Google, Baidu, Naver, Yandex        |
| Deployment   | Docker-ready / Localhost compatible |

---

## 📊 Results

| Algorithm               | Before (Male/Female) | After (Male/Female) | Improvement            |
|-------------------------|----------------------|----------------------|-------------------------|
| Epsilon-Greedy          | 13 / 7               | 8 / 12               | Broke male clustering   |
| Relevance-Aware Swapping| 9 / 11               | 8 / 12               | Balanced with relevance |
| Fairness-Greedy         | 11 / 9               | 10 / 10              | Aligned to ground truth |

All three algorithms demonstrated effectiveness in mitigating bias. **Fairness-Greedy** produced the most balanced and realistic distributions.

---

## 📚 Academic Contribution

This project was submitted as a **conference paper** in ACM-style format for the **CS516 – Fairness and Ethics in AI** course at the University of Illinois at Chicago.

- 📄 Title: *Mitigating Gender Bias in Search Engines*
- 🧑‍💻 Authors: Vishal Goud Mogili, Hemalatha Ningappa Kondakundi, Niketan Doddamani
- 🏫 Institution: University of Illinois at Chicago (UIC)

---

## 🧪 Installation & Running Locally

### ✅ Prerequisites

- Python 3.8+
- Node.js 14+
- Flask
- React
- Caffe Model: `res10_300x300_ssd_iter_140000_fp16.caffemodel`

### 🛠 Setup Instructions

```bash
# Clone the repo
git clone https://github.com/yourusername/gender-bias-search.git
cd gender-bias-search

# Backend Setup
cd backend
pip install -r requirements.txt
python app.py

# Frontend Setup
cd ../frontend
npm install
npm start