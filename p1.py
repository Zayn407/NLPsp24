import pandas as pd
import numpy as np
import nltk
import spacy
from spellchecker import SpellChecker
from spacy.symbols import nsubj, VERB
from collections import Counter

low_essay = "No, i don’t agree with the best way to travel is in a group led. I think in this way they will have many probelme. Firt of all, the group led will be not agree together each one want be the led. Second, when they travel they will be fighting all the time. also, they will not listine to each. n the other hand, when you travel with a group wich has one led, they will be better than onather way for severl reasons. First, all the travels will be nice and specifictly . Next, many people like travel with agroup by one led. Finally, i don’t agree."
high_essay = "I would really prefer to travel on my own with plenty on time, but who wouldn’t? Unfor- tunately that is not always possible. It is always nicer to walk looking around at the same time, steping by little shops and cafes, talking to people, asking for directions, going to the places you choose to go to and discovering everything on your own. I think that is the travel ideal for many of us, but we usually have a hard time on finding the time to do it that way, and instead make plans with too many destinations all at once in a small schedule."


# (a) Length of the essay
def score_length(essay):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(essay)
    num_sentences = len(list(doc.sents))
    average_length_high = 20  # 假设高分作文平均句子数
    average_length_low = 10   # 假设低分作文平均句子数

    # 根据句子数量相对于高低平均值的位置分配得分
    if num_sentences < average_length_low:
        return 1
    elif num_sentences < average_length_high:
        return (num_sentences - average_length_low) / (average_length_high - average_length_low) * 4 + 1
    else:
        return 5


# (b) Spelling mistakes
def score_spelling(essay):
    spell = SpellChecker()
    # 分词并检查拼写
    misspelled = spell.unknown(essay.split())
    num_errors = len(misspelled)
    # 以拼写错误的数量来确定得分，错误越多得分越高，这通常不是期望的评分逻辑
    score = min(num_errors*0.1, 4)
    return round(score, 2)
# (c) Syntax/Grammar
#     
#     (i) Subject-Verb agreement 
def score_subject_verb_agreement(essay):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(essay)
    agreement_errors = 0
    
    for token in doc:
        if token.pos == VERB:
            subjects = [child for child in token.children if child.dep == nsubj]
            for subject in subjects:
                # 单数主语应该对应单数动词形式
                if subject.tag_ in ['NN', 'NNP'] and token.tag_ != 'VBZ':
                    agreement_errors += 1
                # 复数主语应该对应复数动词形式
                elif subject.tag_ in ['NNS', 'NNPS'] and token.tag_ != 'VBP':
                    agreement_errors += 1

    # 根据作文中总句子数量和错误数来计算得分，得分范围在1到5之间
    num_sentences = len(list(doc.sents))
    # 如果有错误，每个错误会根据句子总数减少得分，但分数至少为1
    score = max(5 - (agreement_errors * 5 / num_sentences if num_sentences > 0 else 0), 1)
    return round(score, 2)  # 保留两位小数
#     (ii) Verb tense / missing verb / extra verb 
def score_verb_usage(essay):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(essay)
    verb_errors = 0
    
    for sentence in doc.sents:
        has_aux = False
        has_main_verb = False
        for token in sentence:
            # 检测助动词
            if token.tag_ == "MD":
                has_aux = True
            # 检测主要动词
            if token.pos_ == "VERB" and token.dep_ not in ["aux", "auxpass"]:
                has_main_verb = True

            # 如果存在助动词，但没有主要动词或另一个助动词，则计为错误
            if has_aux and not has_main_verb and token.pos_ != "VERB":
                verb_errors += 1

            # 如果句子结束但缺少主要动词，也计为错误
            if token == sentence[-1] and not has_main_verb:
                verb_errors += 1

            # 这里可以添加更多的检测逻辑，例如检测时态错误等

    num_sentences = len(list(doc.sents))
    # 错误转换为得分，错误越多得分越低
    score = max(5 - (verb_errors * 2 / num_sentences if num_sentences > 0 else 0), 1)
    return round(score, 2)  # 保留两位小数
#     (iii) Sentence formation:
def score_sentence_formation(essay):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(essay)
    total_sentences = len(list(doc.sents))
    sentence_errors = 0
    
    for sentence in doc.sents:
        has_nsubj = False
        has_dobj = False
        for token in sentence:
            # 检查主语
            if token.dep_ == "nsubj":
                has_nsubj = True
            # 检查宾语
            if token.dep_ == "dobj":
                has_dobj = True
        
        # 如果句子缺少主语或宾语，计为错误
        if not has_nsubj or not has_dobj:
            sentence_errors += 1

        # 这里可以添加更多的检查，例如句子是否以适当的标点结尾

    # 计算得分，错误越多得分越低
    score = max(5 - (sentence_errors * 5 / total_sentences if total_sentences > 0 else 0), 1)
    return round(score, 2)
# (d) Semantics (meaning) / Pragmatics (quality at the paragraph/document level):

#     (i) Does the essay answer the question / address the topic? we can use word embeddings.
def score_semantic_relevance(essay, topic):
    nlp = spacy.load("en_core_web_lg")  # 加载包含词向量的大模型
    doc = nlp(essay)
    topic_doc = nlp(topic)
    
    # 计算作文和主题之间的相似性
    similarity = doc.similarity(topic_doc)
    
    # 将相似性得分转换为1到5的评分
    score = 1 + 4 * similarity  # 假设相似性是0到1之间，将其映射到1到5的评分
    return round(score, 2)

#     (ii) Is the essay coherent? We will use a simple algorithm for reference resolution
def score_coherence(essay):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(essay)
    coherence_issues = 0
    transition_words = {'however', 'therefore', 'furthermore', 'consequently', 'then', 'next', 'moreover', 'likewise', 'additionally', 'finally'}
    transition_counts = Counter()

    # 遍历每个句子
    for sentence in doc.sents:
        # 遍历句子中的每个代词
        for token in sentence:
            if token.pos_ == 'PRON':
                # 简化的检查：如果代词的前一个词不是名词或专有名词，可能存在连贯性问题
                prev_token = doc[token.i - 1] if token.i > 0 else None
                if not (prev_token and prev_token.pos_ in ['NOUN', 'PROPN']):
                    coherence_issues += 1
        
        # 统计转折词的使用情况
        transition_counts.update(token.text.lower() for token in sentence if token.text.lower() in transition_words)

    # 如果转折词使用次数太少或太多，视为连贯性问题
    if transition_counts and (max(transition_counts.values()) < 2 or sum(transition_counts.values()) > len(list(doc.sents)) / 2):
        coherence_issues += 1

    total_sentences = len(list(doc.sents))
    # 计算得分，问题越多得分越低
    score = max(5 - (coherence_issues * 5 / total_sentences if total_sentences > 0 else 0), 1)
    return round(score, 2)  # 保留两位小数

# ## total score
# 
def calculate_final_score(essay, topic):
    # 假设以下函数返回1到5的得分
    a = score_length(essay)
    b = score_spelling(essay)  # 这个是0到4的得分，我们可能需要转换为1到5的等级
    ci = score_subject_verb_agreement(essay)
    cii = score_verb_usage(essay)  # 我们假设ci和cii是分开的函数，您可能需要相应调整
    ciii = score_sentence_formation(essay)
    di = score_semantic_relevance(essay, topic)
    dii = score_coherence(essay)  # 假设您已经有一个函数来评估作文的连贯性，这里暂时用1代替

    
    # 按照给定的公式计算最终得分
    final_score = 2 * a - b + ci + cii + 2 * ciii + 3 * di + 2 * dii

    return final_score


import pandas as pd
import numpy as np
import nltk
import spacy
from spellchecker import SpellChecker
from spacy.symbols import nsubj, VERB
from collections import Counter

#############################################################################
def main():
    while True:
        # Prompt the user to input the name of the essay file or type 'quit' to exit
        essay_filename = input("Please enter the name of the essay file (including .txt extension) or type 'quit' to exit: ")

        if essay_filename.lower() == 'quit':
            break

        topic = input("Please enter the topic of the essay: ")

        # Read the essay file
        try:
            with open(essay_filename, 'r') as file:
                essay = file.read()
        except Exception as e:
            print(f"Error reading the file: {e}")
            continue  # Skip to the next iteration if there is an error

        # Calculate and print the scores
        print("\nCalculating scores...")
        # Calculate each score
        a = score_length(essay)
        b = score_spelling(essay)  # Note that this score is on a 0 to 4 scale
        ci = score_subject_verb_agreement(essay)
        cii = score_verb_usage(essay)
        ciii = score_sentence_formation(essay)
        di = score_semantic_relevance(essay, topic)
        dii = score_coherence(essay)

        # Print the scores
        print(f"\nLength score (a): {a}")
        print(f"Spelling score (b): {b} (on a scale of 0 to 4)")
        print(f"Subject-Verb agreement score (c.i): {ci}")
        print(f"Verb tense usage score (c.ii): {cii}")
        print(f"Sentence formation score (c.iii): {ciii}")
        print(f"Semantic relevance score (d.i): {di}")
        print(f"Coherence score (d.ii): {dii}")
        final_score = calculate_final_score(essay, topic)
        print(f"Final score: {final_score}\n")

if __name__ == "__main__":
    main()

