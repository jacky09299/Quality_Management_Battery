import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import jieba
import re
from collections import Counter
import os

# Set font for Chinese characters
import matplotlib.font_manager as font_manager
try:
    font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"Font error: {e}")

output_dir = '/home/jacky/文件/GitHub/Quality_Management_Battery/VOC/output_analysis'
os.makedirs(output_dir, exist_ok=True)

# 1. Load Data
df_app = pd.read_csv('oloo_appstore_reviews.csv')
df_app['platform'] = 'App Store'

df_play = pd.read_csv('oloo_play_reviews_all.csv')
df_play['platform'] = 'Google Play'

# 2. Merge Data
df = pd.concat([df_app, df_play], ignore_index=True)
df['date'] = pd.to_datetime(df['date'])
df['year_month'] = df['date'].dt.to_period('M').astype(str)

# 3. Descriptive & Trend Analysis
# Rating Distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='rating', hue='platform', palette='Set2')
plt.title('Rating Distribution by Platform')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.savefig(f'{output_dir}/rating_dist.png', bbox_inches='tight')
plt.close()

# Monthly Trend
monthly_trend = df.groupby('year_month').agg({'rating': 'mean', 'review': 'count'}).reset_index()
plt.figure(figsize=(12, 6))
ax1 = sns.barplot(data=monthly_trend, x='year_month', y='review', color='lightblue')
ax2 = ax1.twinx()
sns.lineplot(data=monthly_trend, x='year_month', y='rating', color='red', marker='o', ax=ax2)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
ax1.set_ylabel('Review Count (Bars)')
ax2.set_ylabel('Average Rating (Line)')
plt.title('Monthly Review Count and Average Rating')
plt.savefig(f'{output_dir}/monthly_trend.png', bbox_inches='tight')
plt.close()

# Platform Average
platform_stats = df.groupby('platform')['rating'].agg(['mean', 'count', 'std']).round(2)
platform_stats.to_csv(f'{output_dir}/platform_stats.csv')

# 4. Version Analysis
top_versions = df['version'].value_counts().nlargest(10).index
version_data = df[df['version'].isin(top_versions)]
plt.figure(figsize=(10, 6))
sns.boxplot(data=version_data, x='version', y='rating', palette='pastel', order=top_versions)
plt.title('Rating Distribution by Top 10 Versions')
plt.xticks(rotation=45)
plt.savefig(f'{output_dir}/version_rating.png', bbox_inches='tight')
plt.close()

version_stats = version_data.groupby('version')['rating'].agg(['mean', 'count']).sort_values('count', ascending=False).round(2)
version_stats.to_csv(f'{output_dir}/version_stats.csv')

# 5. Text Mining for Negative Reviews (1-2 stars)
negative_reviews = df[df['rating'] <= 2]['review'].dropna().astype(str)

# Basic stopwords
stopwords = set(['的', '了', '是', '我', '這', '也', '不', '就', '都', '有', '在', '很', '沒', '啊', '可以', '一個', '但是', '什麼', '還是', '就是', '連', '因為', '所以', '一直', '沒有', '這個', '我們', '如果', '到', '說', '要', '去', '跟', '給', '上', '還', '然後', '根本', '而且', '不會', '一樣', '知道', '太', '怎麼', '真的', '不能', '覺得', '才', '被'])
words = []
for review in negative_reviews:
    # Remove punctuation and english letters
    review = re.sub(r'[^\w\s]', '', review)
    review = re.sub(r'[a-zA-Z0-9]', '', review)
    for word in jieba.cut(review):
        if len(word) > 1 and word not in stopwords:
            words.append(word)

word_counts = Counter(words)
top_20_words = word_counts.most_common(20)
top_words_df = pd.DataFrame(top_20_words, columns=['Word', 'Count'])
top_words_df.to_csv(f'{output_dir}/top_negative_words.csv', index=False)

# Need to avoid font issue on server for text, so just save csv and print
print("Platform Stats:")
print(platform_stats)
print("\nVersion Stats (Top 10):")
print(version_stats)
print("\nTop 20 Negative Keywords:")
print(top_words_df)
print("\nAnalysis complete. Visualizations saved in output_analysis directory.")
