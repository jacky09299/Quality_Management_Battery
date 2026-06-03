import pandas as pd
import jieba
import re
from collections import Counter
import os

# 1. Load Data
df_app = pd.read_csv('oloo_appstore_reviews.csv')
df_play = pd.read_csv('oloo_play_reviews_all.csv')
df = pd.concat([df_app, df_play], ignore_index=True)

# 2. Text Preprocessing
stopwords = set(['的', '了', '是', '我', '這', '也', '不', '就', '都', '有', '在', '很', '沒', '啊', '可以', '一個', '但是', '什麼', '還是', '就是', '連', '因為', '所以', '一直', '沒有', '這個', '我們', '如果', '到', '說', '要', '去', '跟', '給', '上', '還', '然後', '根本', '而且', '不會', '一樣', '知道', '太', '怎麼', '真的', '不能', '覺得', '才', '被', '然後', '吧', '嗎', '呢', '而已', '雖然', '只是', '已經', '又', '只'])

def extract_ngrams(texts, n=2):
    ngrams = []
    for text in texts:
        text = str(text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'[a-zA-Z0-9]', '', text)
        words = [w for w in jieba.cut(text) if len(w) > 1 and w not in stopwords]
        for i in range(len(words)-n+1):
            ngram = "".join(words[i:i+n])
            # Keep track of original words for matching later
            ngrams.append((ngram, tuple(words[i:i+n])))
    
    # Count the combined string
    counter = Counter([item[0] for item in ngrams])
    top_ngrams = counter.most_common(15)
    
    # Map back to original words
    result = []
    for ngram_str, count in top_ngrams:
        # Find the original tuple
        for item in ngrams:
            if item[0] == ngram_str:
                result.append((ngram_str, count, item[1]))
                break
    return result

# Split into Positive and Negative
df_neg = df[df['rating'] <= 2]['review'].dropna()
df_pos = df[df['rating'] >= 4]['review'].dropna()

# Get frequent bi-grams
neg_bigrams = extract_ngrams(df_neg, 2)
pos_bigrams = extract_ngrams(df_pos, 2)

# Generate Markdown
md_content = "# 深度評論文字探勘：好與壞的具體原因 (情境還原)\n\n"
md_content += "您說得對，單看單一詞彙（例如：「問題」、「還車」）很難感受到使用者具體遇到了什麼狀況。因此我重新分析了「**連續詞彙（Bi-grams）**」，並直接從資料中抓取了**對應的真實評論對話**，讓您一眼就能看出好與壞的脈絡。\n\n"

md_content += "## 🔴 負評具體痛點 (1-2 顆星)\n"
md_content += "以下是使用者給予低分時，最常連續抱怨的具體情境：\n\n"

for ngram_str, count, words_tuple in neg_bigrams:
    md_content += f"### ⚠️ 痛點標籤：【{ngram_str}】 (共同出現 {count} 次)\n"
    
    examples = []
    for text in df_neg:
        text_str = str(text)
        if all(w in text_str for w in words_tuple):
            examples.append(text_str.replace('\n', ' '))
            if len(examples) == 3:
                break
    
    for i, ex in enumerate(examples, 1):
        md_content += f"> 🗣️ **評論 {i}：** \"{ex}\"\n"
    md_content += "\n"

md_content += "---\n\n"

md_content += "## 🟢 正評亮點分析 (4-5 顆星)\n"
md_content += "另一方面，這裡整理了給予高分的使用者，最滿意的具體部分：\n\n"

if not pos_bigrams:
    md_content += "正評數量較少或缺乏共通的連續詞彙。\n"

for ngram_str, count, words_tuple in pos_bigrams:
    md_content += f"### ✨ 滿意標籤：【{ngram_str}】 (共同出現 {count} 次)\n"
    
    examples = []
    for text in df_pos:
        text_str = str(text)
        if all(w in text_str for w in words_tuple):
            examples.append(text_str.replace('\n', ' '))
            if len(examples) == 3:
                break
    
    for i, ex in enumerate(examples, 1):
        md_content += f"> 🗣️ **評論 {i}：** \"{ex}\"\n"
    md_content += "\n"

artifact_path = "/home/jacky/.gemini/antigravity-cli/brain/9a65d5bf-2f11-489c-8ab3-a76b9d5eeca6/Advanced_VOC_Text_Analysis.md"
with open(artifact_path, "w", encoding="utf-8") as f:
    f.write(md_content)

print(f"Artifact created at {artifact_path}")
