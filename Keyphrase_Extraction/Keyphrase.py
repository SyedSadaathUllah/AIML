from rake_nltk import Rake
from wordcloud import WordCloud
import matplotlib.pyplot as plt

rake = Rake()

input_text = '''
Artificial Intelligence is transforming the world around us. From voice assistants and self-driving cars to healthcare innovations and personalized recommendations, AI is becoming an integral part of our daily lives. As technology advances, understanding and harnessing AI responsibly will shape the future of human progress.
'''

rake.extract_keywords_from_text(input_text)
keywords = rake.get_ranked_phrases()
print(keywords)

wordcloud = WordCloud().generate(' '.join(keywords))
plt.figure(figsize=(10,10))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.savefig("wordcloud.png")
