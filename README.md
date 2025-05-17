# AI Resume Analyzer and Match Maker for Job Seekers 🔍

![Project Banner](https://github.com/znixonz/MantapResume/blob/main/mantap/app/logo/mantap_logo.jpeg?raw=true) 

A smart resume analysis system that leverages NLP and Qwen LLM to evaluate resumes, recommend career paths, and connect candidates with optimal job opportunities through Alibaba Cloud infrastructure.

## 🌟 Key Features

| Feature | Description |
|---------|-------------|
| 📄 **Resume Scoring** | AI-powered resume evaluation with similarity matching against job requirements |
| 🎯 **Job Matching** | Intelligent job recommendations based on skills and experience |
| 📚 **Course Recommendations** | Personalized learning path suggestions for skill gaps |
| 📊 **Analytics Dashboard** | Interactive visualizations of resume metrics and market trends |
| ☁️ **Cloud Integration** | Built on Alibaba Cloud ecosystem for enterprise scalability |

## 🛠️ Tech Stack

**Frontend**  
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)

**Backend**  
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Qwen LLM](https://img.shields.io/badge/Qwen-LLM-important?style=for-the-badge)

**Alibaba Cloud Infrastructure**  
![ApsaraDB](https://img.shields.io/badge/ApsaraDB-PostgreSQL-blue?style=for-the-badge)
![ECS](https://img.shields.io/badge/Elastic_Compute_Service-ECS-orange?style=for-the-badge)
![OSS](https://img.shields.io/badge/Object_Storage-OSS-lightgrey?style=for-the-badge)

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Alibaba Cloud account
- Streamlit environment

### Installation
```bash
# Clone repository
git clone https://github.com/znixonz/MantapResume.git

# Navigate to project directory
cd mantap/app

# Set environment variables
export ALIBABA_API_KEY=<your-api-key>
export ALIBABA_DB_ENDPOINT=<db-endpoint>

# Install dependencies
pip install -r requirements.txt

# Launch application
streamlit run App_new.py
```

## 📐 Architecture

![Screenshot](https://github.com/znixonz/MantapResume/blob/main/img/hack.jpg?raw=true)

## 🧩 Core Components

1. **Resume Processing Engine**
   - PDF/text parsing with NLP
   - Skill extraction and normalization
   - Experience quantification

2. **Qwen LLM Integration**
   - Semantic similarity analysis
   - Job description understanding
   - Career path predictions

3. **Alibaba Cloud Services**
   - ApsaraDB for candidate profiles
   - OSS for resume storage
   - ECS for scalable processing

## Preview

![Screenshot](https://github.com/znixonz/MantapResume/blob/main/img/1.png?raw=true)

![Screenshot](https://github.com/znixonz/MantapResume/blob/main/img/2.png?raw=true)

![Screenshot](https://github.com/znixonz/MantapResume/blob/main/img/3.png?raw=true)

