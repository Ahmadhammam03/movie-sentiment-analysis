# 🎬 Movie Review Sentiment Analysis

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.7+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/NLTK-VADER-green.svg" alt="NLTK VADER">
  <img src="https://img.shields.io/badge/Accuracy-63.67%25-orange.svg" alt="Accuracy">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen.svg" alt="Status">
  [![LinkedIn](https://img.shields.io/badge/-LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ahmad-hammam-1561212b2)
  [![GitHub](https://img.shields.io/badge/-GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Ahmadhammam03)
</div>

<br>

<div align="center">
  <h3>🤖 Professional VADER Sentiment Analysis on IMDb Movie Reviews</h3>
  <p><strong>Comprehensive NLP project analyzing 2,000 movie reviews with advanced error analysis and statistical evaluation</strong></p>
</div>

## 🎯 Project Overview

This project implements **VADER (Valence Aware Dictionary and sEntiment Reasoner)** sentiment analysis on the famous **IMDb Movie Reviews Dataset** containing 2,000 movie reviews. The goal is to automatically classify movie reviews as positive or negative sentiment and evaluate the performance against human-labeled ground truth.

### 🔍 Key Highlights
- **Dataset**: 2,000 IMDb movie reviews (1,000 positive, 1,000 negative)
- **Algorithm**: NLTK VADER Sentiment Intensity Analyzer
- **Performance**: 63.67% accuracy with detailed analysis
- **Challenge**: Understanding complex human semantics in movie reviews
- **Analysis**: 20+ professional visualizations and statistical tests

## 🧠 What Makes This Project Special

Unlike simple text classification, this project tackles the challenging problem of **sentiment analysis in movie reviews**, where:

- Reviews often contain mixed sentiments
- Sarcasm and irony are common
- Context matters more than individual words
- Final judgments may appear only at the end of long reviews

## 📊 Key Results

```
✅ Overall Accuracy: 63.67%
📊 Total Reviews Analyzed: 1,938
✔️ Correct Predictions: 1,234
❌ Incorrect Predictions: 704

📈 Performance Breakdown:
• Precision (Negative): 72.0%
• Precision (Positive): 60.0%
• Recall (Negative): 44.0%
• Recall (Positive): 83.0%
• F1-Score (Macro): 62.0%
```

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.7+
pip install -r requirements.txt
```

### Installation
```bash
# Clone the repository
git clone https://github.com/Ahmadhammam03/movie-sentiment-analysis.git
cd movie-sentiment-analysis

# Create virtual environment
python -m venv sentiment_env
source sentiment_env/bin/activate  # On Windows: sentiment_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('vader_lexicon')"
```

### Run the Analysis
```bash
# Launch Jupyter Notebook
jupyter notebook

# Open and run the enhanced notebook
# -> Sentiment-Analysis.ipynb
```

## 📁 Project Structure

```
movie-sentiment-analysis/
│
├── 📓 Sentiment-Analysis.ipynb                       # Main analysis notebook
├── 📊 data/
│   └── moviereviews.tsv                              # Movie reviews dataset
├── 🖼️ assets/                                        # Generated visualizations
│   ├── sentiment_analysis_banner.png                 # Project banner
│   ├── confusion_matrix.png                          # Confusion matrix visualization
│   ├── sentiment_distribution.png                    # Data distribution plot
│   ├── methodology_flow.png                          # Process flow diagram
│   └── error_examples.png                            # Error analysis examples
├── 📋 requirements.txt                               # Python dependencies
├── 📄 README.md                                      # Project documentation
├── 📜 LICENSE                                        # MIT License
└── 🚫 .gitignore                                     # Git ignore rules
```

## 🔬 Methodology

### 1. Data Preprocessing
- **Data Loading**: Import 2,000 movie reviews from TSV file
- **Data Cleaning**: Remove NaN values and empty strings
- **Quality Assessment**: Analyze text length and characteristics

### 2. VADER Sentiment Analysis
```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize VADER analyzer
sid = SentimentIntensityAnalyzer()

# Generate sentiment scores
scores = sid.polarity_scores(review_text)
# Returns: {'neg': 0.121, 'neu': 0.778, 'pos': 0.101, 'compound': -0.9125}
```

### 3. Performance Evaluation
- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score
- **Error Analysis**: False positives/negatives with examples
- **Statistical Testing**: Mann-Whitney U tests for significance
- **Visualization**: 20+ professional charts and dashboards

## 📈 Key Findings

### 🎯 **VADER Strengths:**
- ✅ High recall for positive reviews (83%) - excellent at finding positive sentiment
- ✅ Good precision for negative reviews (72%) - confident negative predictions
- ✅ Fast processing (500+ reviews/second)
- ✅ No training required - ready-to-use approach
- ✅ Interpretable results with sentiment components

### ⚠️ **VADER Limitations:**
- ❌ 542 false positives - struggles with negative reviews containing positive words
- ❌ Limited context understanding beyond individual words
- ❌ Difficulty with sarcasm, irony, and mixed sentiments
- ❌ Challenges with delayed judgment patterns in movie reviews

### 💡 **Research Insights:**
- Movie reviews exhibit complex sentiment structures not easily captured by lexicon-based methods
- Many reviews discuss positive aspects before delivering negative final judgment
- Context and discourse structure matter more than individual word sentiment
- The decision boundary may need domain-specific optimization

## 🎨 Visualizations

The project includes comprehensive visualizations:

- **Dataset Analysis Dashboard** (4-panel overview)
- **Sentiment Score Analysis** (9-panel comprehensive view)
- **Performance Evaluation Dashboard** (9-panel metrics analysis)
- **Error Analysis Visualizations** (detailed error patterns)
- **Final Summary Dashboard** (key findings and recommendations)

## 🛠️ Technical Stack

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white">
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white">
  <img src="https://img.shields.io/badge/NLTK-154f3c?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white">
  <img src="https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=Jupyter&logoColor=white">
</div>

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **nltk**: Natural Language Processing toolkit with VADER
- **scikit-learn**: Machine learning metrics and evaluation
- **matplotlib/seaborn**: Professional data visualization
- **scipy**: Statistical analysis and significance testing

## 📚 Dataset Information

### Source
- **Original Dataset**: Cornell Movie Review Data v2.0
- **Authors**: Bo Pang and Lillian Lee (2004)
- **Publication**: "A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization Based on Minimum Cuts", ACL 2004
- **Size**: 2,000 movie reviews from IMDb
- **Balance**: Perfectly balanced (1,000 positive, 1,000 negative)

### Citation
```bibtex
@InProceedings{Pang+Lee:04a,
  author = {Bo Pang and Lillian Lee},
  title = {A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization Based on Minimum Cuts},
  booktitle = "Proceedings of the ACL",
  year = 2004
}
```

## 🔮 Future Improvements

### 📈 **Immediate Enhancements**
- Adjust decision threshold based on domain-specific validation
- Implement preprocessing to handle negations more effectively
- Add movie-specific sentiment lexicon expansions
- Consider sentence-level sentiment analysis for longer reviews

### 🚀 **Advanced Approaches**
- Implement transformer-based models (BERT, RoBERTa) for better context understanding
- Use ensemble methods combining VADER with other sentiment approaches
- Develop domain-specific models trained on movie review corpora
- Apply attention mechanisms to identify key sentiment-bearing passages

## Learning Outcomes

### Technical Skills Developed
- **Sentiment Analysis**: Understanding VADER lexicon-based approach
- **Data Preprocessing**: Cleaning and preparing text data for analysis
- **Performance Evaluation**: Using comprehensive classification metrics
- **Statistical Analysis**: Conducting significance tests and error analysis
- **Data Visualization**: Creating professional, publication-ready charts

### Domain Knowledge Gained
- **NLP Challenges**: Real-world complexities in sentiment analysis
- **Human Language**: Understanding context and semantic nuances
- **Model Limitations**: When rule-based systems fall short
- **Research Methods**: Systematic evaluation and error analysis

## 🤝 Contributing

Contributions are welcome! Here's how you can help improve this project:

### 🛠️ **Areas for Contribution**
- Add additional sentiment analysis methods for comparison
- Implement deep learning approaches (BERT, RoBERTa)
- Create interactive web interface for real-time analysis
- Add more comprehensive error analysis categories
- Improve documentation and add more examples

### 📝 **How to Contribute**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

  **Ahmad Hammam**
- GitHub: [@Ahmadhammam03](https://github.com/Ahmadhammam03)
- LinkedIn: [Ahmad Hammam](https://www.linkedin.com/in/ahmad-hammam-1561212b2)

**Skills Demonstrated in This Project:**
- Advanced Python programming and data analysis
- Natural Language Processing with NLTK
- Statistical analysis and hypothesis testing
- Professional data visualization and storytelling
- Research methodology and systematic evaluation
- Technical documentation and communication

## 🙏 Acknowledgments

- **Cornell University** for providing the movie review dataset
- **NLTK Team** for the excellent VADER sentiment analysis tool
- **Open Source Community** for making advanced NLP accessible to everyone

## 📊 Project Stats

<div align="center">
  <img src="https://img.shields.io/github/stars/Ahmadhammam03/movie-sentiment-analysis?style=social">
  <img src="https://img.shields.io/github/forks/Ahmadhammam03/movie-sentiment-analysis?style=social">
  <img src="https://img.shields.io/github/watchers/Ahmadhammam03/movie-sentiment-analysis?style=social">
</div>

---

<div align="center">
  <p><strong>⭐ If you found this project helpful, please consider giving it a star! ⭐</strong></p>
  
  **🎬 "In the world of sentiment analysis, context is king and VADER is a worthy knight!" 🎭**
</div>
