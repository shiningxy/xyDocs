# 人工智能：现代方法 第四版

## 准备

官方习题：https://aimacode.github.io/aima-exercises/

官方代码：https://github.com/aimacode/aima-python

官方全书彩图与伪代码：http://aima.cs.berkeley.edu/figures.pdf

中文全书彩图与伪代码：https://box.lenovo.com/l/r1rKtK

英文版官方书籍PDF：https://drive.google.com/file/d/137KFeICM-Wrou6hpI3OxU-l-l-wky103/view?usp=sharing

中文版前三章PDF: https://drive.google.com/file/d/1-2uujQLbz7kBTk2xlpvoAgcJUAzkXmzf/view?usp=sharing

## 第一章习题

> Part I Artificial Intelligence: 1. Introduction

### Question
:question: 1. 用您自己的话来定义：（a）智能，（b）人工智能，（c）智能体，（d）理性，（e）逻辑推理。
> Define in your own words: (a) intelligence, (b) artificial intelligence, (c) agent, (d) rationality, (e) logical reasoning.

:exclamation:（a）智能：是智力和能力的总称，中国古代思想家一般把智与能看做是两个相对独立的概念。也有不少思想家把二者结合起来作为一个整体看待。

（b）人工智能：是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器，该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。它研究的一个主要目标是使机器能够胜任一些通常需要人类智能才能完成的复杂工作。

（c）智能体：具有智能的实体，以云为基础，以AI为核心，构建一个立体感知、全域协同、精准判断、持续进化、开放的智能系统。或者简单说能够采取行动，能够自主运行、感知环境、长期持续存在、适应变化以及制定和实现目标。

（d）理性：指人在正常思维状态下时为了获得预期结果，有自信与勇气冷静地面对现状，并快速全面了解现实分析出多种可行性方案，再判断出最佳方案且对其有效执行的能力。理性是基于现有的理论，通过合理的逻辑推导得到确定的结果，

（e）逻辑推理：从一般性的前提出发，通过推导，得出具体陈述或个别结论的过程。逻辑推理的种类按推理过程的思维方向划分，一类是从特殊到一般的推理，推理形式主要有归纳、类比，另一类是从一般到特殊的推理,推理形式主要有演绎。

---

:question: 2. 阅读图灵关于 AI Turing:1950 的原始论文。在论文中，他讨论了对他提出的企业和他的智能测试的几点反驳意见。哪些反对意见仍然有分量？他的反驳是否有效？你能想到自从他写这篇论文以来，事态发展引发的新的反对意见吗？在论文中，他预测到2000年，计算机将有30%的几率通过五分钟的图灵测试，而不需要熟练的询问器。你认为今天电脑有什么机会？再过50年？
> Read Turing’s original paper on AI Turing:1950 .In the paper, he discusses several objections to his proposed enterprise and his test for intelligence. Which objections still carry weight? Are his refutations valid? Can you think of new objections arising from developments since he wrote the paper? In the paper, he predicts that, by the year 2000, a computer will have a 30% chance of passing a five-minute Turing Test with an unskilled interrogator. What chance do you think a computer would have today? In another 50 years?

:exclamation:图灵的九条反驳意见至今仍有分量且有效，他们分别是：神学论点，“鸵鸟”式论点，数学论点，意识论点，种种能力限制的论点，创新论点，神经系统连续性论点，行为变通性论点，超感知论点。今天的电脑拥有更加强大的算力，比以前更加优秀的深度学习，强化学习等方法。因此如今的电脑在很多方面都能做的比人类还好，比如图像识别的正确率方面。如果再过50年，很难想象，我认为机器的发展速度和学习能力比人类强太多太多了，也许会有颠覆认知东西产生。

---

:question: 3. 每年的罗布纳奖(Loebner Prize)都会颁发给最接近通过图灵测试的程序。调研最新的罗布纳奖得主。它使用什么技术？它如何推动人工智能的发展？ 注：勒布纳奖已在2020年停止颁发。
> Every year the Loebner Prize is awarded to the program that comes closest to passing a version of the Turing Test. Research and report on the latest winner of the Loebner prize. What techniques does it use? How does it advance the state of the art in AI?

:exclamation:Stephen Worswick 是目前为止最新的罗布纳奖得主，也获得最多该奖的人。他所开发的聊天机器人 Mitsuku，曾在 2013 年、2016 年、2017 年、2018 年和 2019 年获奖。Mitsuku 是一个非常像人的聊天机器人，它模仿英格兰北部一位 18 岁女性的个性。任何有网络连接的人都可以自由地与 Mitsuku 聊天，问她任何问题，它都会回答。它在模仿人类反应方面做得很好，能够识别现代口语、计算机缩写（如 LOL）和时事。询问她最喜欢的足球队，她会“愉快地”告诉您有关利兹联队的信息。有趣的是，“终结者”是她最喜欢的电影。虽然 Mitsuku 逐年改进，但聊天机器人和文本另一端的人类之间仍然存在明显的差异。Mitsuku 很难识别常见的拼写错误或字母互换。即使它努力与互动的人继续对话，对话也存在一个明显的截止点。

它使用 AIML (Artificial Intelligence Markup Language) 来识别用户输入的关键词和语法，并回复相应的回答。AIML 是一种基于模式匹配的语言，用于定义人工智能与人类之间的对话。它允许开发人员使用类似于编程语言的语法来定义人机对话的模式。 例如，如果有人问“你能吃下一个房子吗？”，它会查询“房子”的性质。发现“made_from”的值是“brick”，就会回答“不能”。

Mitsuku 作为一个高度可定制的聊天机器人，它能够与人类进行多种类型的对话，并且不断改进。这也推动了人工智能在自然语言处理和对话系统方面的发展。 提高自然语言理解能力： Mitsuku 使用 AIML 技术来识别和理解人类语言，这有助于提高 AI 的自然语言理解能力。 提高对话系统的可用性： Mitsuku 可以进行各种类型的对话，并且可以根据用户的需求进行自定义，这有助于提高 AI 对话系统的可用性。 提高对话系统的交互性： Mitsuku 能够与人类进行自然的对话，这有助于提高 AI 对话系统的交互性。

---

:question: 4. 反射动作（例如从热炉中退缩）是否合理？他们智能吗？
> Are reflex actions (such as flinching from a hot stove) rational? Are they intelligent?

:exclamation:反射动作是合理的。反射动作通常比经过深思熟虑后采取的较慢的动作更为成功，它属于理性行为的一种方式，而理性行为是理性智能体方法，它也是智能的。

---

:question: 5. 有一些众所周知的问题是计算机难以解决的，还有一些问题是无法确定的。这是否意味着人工智能是不行的？
> There are well-known classes of problems that are intractably difficult for computers, and other classes that are provably undecidable. Does this mean that AI is impossible?

:exclamation:虽然有一些众所周知的问题是计算机难以解决的，但从人工智能的诞生（1943-1956），起步发展期（1956-1969），反思发展期（1966-1973），应用发展期-专家系统（1969-1986），神经网络的回归（1986-现在），概率推理和机器学习（1987-现在），大数据（2001-现在），深度学习（2011-现在）。经历这些阶段之后产生的新兴方向，比如自动驾驶、腿足式机器人，自动规划和调度，机器翻译，语音识别，推荐系统，博弈，图像理解，医学，气候科学等等这些，表明人工智能在特定方向进步的速度比人类还快。按照这种发展与进步的速度，人工智能迟早可以解决那些众所周知的问题的。所以这并不意味着人工智能不行。

---

:question: 6. 假设我们扩展了Evans的SYSTEM程序，使其在标准智商测试中可以获得200分。那么我们会有一个比人类更聪明的程序吗？
> Suppose we extend Evans’s SYSTEM program so that it can score 200 on a standard IQ test. Would we then have a program more intelligent than a human? Explain.

:exclamation:目前还不能做到一个比人类更聪明的程序，即使其在标准智商测试能获得200分。目前人工智能也只能在单一方面或者某些方面有较出色的表现，在综合方面的学习与表现是远不如人类的。智商测试虽然能达到200分，那也只是在智商测试这一方面很卓越，在其他方面并不能表现都很出色。

--- 

:question: 7. sea slug Aplysis 的神经结构得到了广泛的研究(首先是由诺贝尔奖获得者埃里克·坎德尔(Eric Kandel)进行的)，因为它只有大约2万个神经元，其中大多数都很大，很容易操纵。假设Aplysis神经元的周期时间与人类神经元大致相同，那么就每秒内存更新而言，与图1.3中描述的高端计算机相比，其计算能力如何?

> The neural structure of the sea slug Aplysis has been widely studied (first by Nobel Laureate Eric Kandel) because it has only about 20,000 neurons, most of them large and easily manipulated. Assuming that the cycle time for an Aplysis neuron is roughly the same as for a human neuron, how does the computational power, in terms of memory updates per second, compare with the high-end computer described in (Figure 1.3)?

:exclamation:Aplysis神经结构的计算能力相对于高端计算机来说要低得多。Aplysis只有约2万个神经元，而高端计算机可以拥有数十亿个处理器。因此，Aplysis的计算能力要远远低于高端计算机。

---

:question: 8. 自省——对一个人内心想法的报告——怎么会不准确呢?我的想法会不会是错的?请讨论。
> How could introspection—reporting on one’s inner thoughts—be inaccurate? Could I be wrong about what I’m thinking? Discuss.

:exclamation: 所谓自省，指一个人内心想法进行思考的过程，是不准确的。原因是自我反省是具有偏见的，这些偏见来自人类生活的方方面面，有对事物客观认知错误的认知偏差，有为了满足大脑趋利避害需求的自我欺骗偏差，有受限于大脑知识容量而忘记部分信息的记忆偏差，也有社会地位偏差，比如社会不同阶级的人无法以理性的角度看同一件事情。综上，我认为人的自我反省是不准确的，但可以在反省的过程中借助工具和他人的力量做到尽量准确。

---

:question: **Question 9:** 以下计算机系统实例是否是人工智能的例子:
* 超市条码扫描器。
* 网络搜索引擎。
* 语音激活的电话菜单。
* 对网络状态作出动态反应的互联网路由算法。

> To what extent are the following computer systems instances of artificial intelligence: - Supermarket bar code scanners. - Web search engines. - Voice-activated telephone menus. - Internet routing algorithms that respond dynamically to the state of the network.

:exclamation: **Answer 9:**
- 超市条码扫描器：不被认为是人工智能的应用，他们只是用来执行特定的程序，将商品上的二维码匹配到固定的信息上，不被认为具有泛化到其他任务的潜力 
- 网络搜索引擎：搜索系统被认为是人工智能的实例，因为它们为根据使用者的查询返回相关结果。它们的设计中包含根据使用者的反馈来学习和优化内部的算法参数，从而改进搜索结果。 
- 语音激活的电话菜单：语音激活系统被认为是人工智能的实例，因为它们具有识别和回应口头命令的能力，可以理解自然语言。 
- 对网络状态作出动态反应的互联网路由算法： 现代的互联网路由算法被认为是人工智能的实例，因为它们能够根据实时网络信息做出决策，并根据反馈不断变化来更好的适应当前的网络条件。

---

:question: **Question 10.** 以下计算机系统在多大程度上是人工智能的实例:
* 超市的条形码扫描器。
* 语音激活的电话菜单。
* Microsoft Word中的拼写和语法纠正功能。
* 对网络状态作出动态反应的互联网路由算法

> To what extent are the following computer systems instances of artificial intelligence: - Supermarket bar code scanners. - Voice-activated telephone menus. - Spelling and grammar correction features in Microsoft Word. - Internet routing algorithms that respond dynamically to the state of the network.

:exclamation: **Answer 10:**
- 超市的条形码扫描器：同1.9 
- 语音激活的电话菜单：同1.9 
- Microsoft Word中的拼写和语法纠正功能：被认为是人工智能算法的实例，可以理解用户输入语句的意思，并找出错误，可以随着反馈更新迭代自己的参数，越变越好，具有学习能力。 
- 对网络状态作出动态反应的互联网路由算法：同1.9