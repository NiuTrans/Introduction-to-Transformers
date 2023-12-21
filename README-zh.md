# 自然语言处理视角下的Transformer模型

Transformer模型已经在自然语言处理中占据主导地位，并成为了当今人工智能领域最有影响的深度学习模型之一。本项目（详见论文：[Introduction to Transformers: an NLP Perspective](https://arxiv.org/abs/2311.17633)）从自然语言处理（NLP）的视角介绍了Transformer模型，并展示了该模型最新进展中所使用的关键技术。鉴于Transformer模型和相关的深度学习技术正在以前所未见的方式高速发展，这里无法深入所有模型细节或覆盖所有技术领域。相反，我们专注于那些有助于深入理解Transformer的基本概念，并总结了影响这一领域的关键思想，从而可以洞察该模型的优势和局限。

*阅读上述论文需要具备线性代数和概率论基础知识，同时有一定的了解深度学习的基本技术。但即便是不具备专业背景知识的读者，可以跳过需要特殊背景知识的部分，也能对Transformer有一个大致的理解。*

*本项目的英文版网页：https://github.com/NiuTrans/Introduction-to-Transformers*


## Transformer总览

自从2017年Transformer问世，以Transformer架构为基础的模型、算法层出不穷。放眼整个人工智能领域，已经发表了数量巨大的以Transformer为主题的论文。对所有这些论文进行深入分析似乎是一件不太可能的工作。

本项目从自然语言处理的视角出发，尝试描述Transformer模型的总体概览和众多可行方案中的一种分类方法如下图，并结合论文对每个模块进行详细地讲解，最后提供了一份简短的Transformer相关论文列表以供参考，希望对NLP初学者有所帮助、对NLP从业者有所启发。

![transformer-pic](figures/figure-transformer-roadmap.jpg#pic_center)

可以从如下几个方面了解Transformer：

**Transformer基础**：Transformer作为序列建模的标准框架，为NLP领域相关研究带来了巨大变革，但实际上Transformer并不是一个“全新”模型，而是由许多早期深度学习领域提出的高性能组件（包括词嵌入和位置嵌入、注意力机制、残差连接、层归一化等）整合设计而成的“独特”模型。例如，Transformer中多头注意力与点积QKV注意力的结合，以及层归一化和残差连接的融入，形成了独特的神经网络模块（自注意力子层）。因此，详细了解每个关键组件的工作原理，对于深入理解其如何在Transformer架构中发挥作用是至关重要的。

**注意力模型**：为什么Transformer在NLP领域的各项任务中均取得了引人注目的成果？实际上很大程度归因于其核心技术———注意力机制，大量实验验证在序列建模中使用多头自注意力可以显著提升模型性能。由此引发了众多学者对注意力机制的广泛研究，这里抛砖引玉，提出几个值得关注的研究方向供大家参考：1）修改QKV注意力和多头注意力的形式，例如在自注意力子层中添加新组件，以适应特定任务，提高模型性能；2）将先验知识融入到注意力模型的设计，例如将句法解析输入基于Transformer的系统，以实现模型在语言学上的可解释性；3）开发高效的注意力模型，例如使用稀疏或循环模型对自注意力机制进行近似，简化其结构。除了探索更强大、更高效的注意力模型之外，研究人员发现了十分有趣的现象，尽管模型的训练目标不是表示学习到的知识，但多头自注意力可以学习到语言的底层结构。关于注意力机制的有效改进方法、有趣现象和具体分析等等推荐阅读论文第4&5节，以及后续论文列表中提供的“注意力”相关经典文献。

**词嵌入和位置嵌入**：Transformer架构的底层模块中，词嵌入和位置嵌入扮演着关键角色。在输入阶段，Transformer用词嵌入表示每个输入词，用位置嵌入帮助注意力机制理解词在序列中的顺序。词嵌入可以借助已经训练好的Word2Vec或者GloVe嵌入，也可以从头开始训练学习。其中，分词会直接影响模型输出的token数量以及词嵌入的学习难度等，是词嵌入模块的关键问题。当模型要处理的序列长度明显长于训练数据时，位置嵌入可以通过用旋转位置嵌入替换正弦位置嵌入，或者简单地用位置标量缩放注意力权重来实现外推。尽管位置嵌入是一个普遍问题，但实际上大部分研究更集中于改进Transformer。

**训练和模型扩展**：在深度学习时代，强大的系统通常受益于大型神经网络，这对于增强Transformer性能同样适用。显然，我们可以通过堆叠更多的网络层或者扩大每一层的表示大小来增加模型容量，大量实验表明更深、更宽的Transformer模型性能始终优异于小模型。然而，训练大规模模型不可避免地存在计算资源方面的挑战，尤其是在大量数据应用梯度下降算法的情况下。一种通用的工程解决方案是在计算机集群上进行分布式训练，训练方法的改进也同步影响着模型架构的设计，例如稀疏专家模型可以简化带有分布式参数的训练，为许多基于Transformer的系统提供了基础。随着Transformer系列模型的规模不断扩增，研究人员逐渐探索出大型神经网络的扩展规律：模型性能与模型规模、训练数据量和训练成本三者之间存在紧密的相关性，当模型规模扩大到一定量级随之产生了“涌现”的有趣现象。在最近的NLP研究中，具备“涌现能力”被视为强大语言模型的先决条件之一。

**高效模型**：关于模型效率，针对不同的问题有不同的定义。例如，当内存受限时，目标是节省内存占用；当系统延迟时，目标是提高运行速度。在实际应用当中，我们通常寻求多个目标之间的平衡，从而实现系统的整体效率提升。在Transformer的背景下，大多数优化方法基于改进注意力模型，实现减少处理长序列时的内存占用以及减少计算量降低延迟。还有一些与模型架构无关的优化方法，包括但不限于条件计算、知识蒸馏、结构化剪枝和序列压缩。也可以从计算机架构的角度出发，针对Transformer的不同组件，例如计算密集型的编码过程和IO密集型解码过程，采用不同的优化方法。

**推理**：在序列生成任务中，推理是研究人员重点关注的问题之一。在自然语言处理过程中，模型可以应用如广度优先搜索、深度优先搜索和A*搜索等算法，在数十个甚至数百个tokens的序列空间中寻找“最佳”假设。在实际应用当中，搜索系统的效率是一个重要考虑因素，因此优化搜索算法十分必要。这里简单列举几个优化方案：1）将机器翻译和自动语音识别（ASR）中的搜索优化算法直接应用于文本生成模型；2）优化传统解码方法；3）将高效注意力机制部署于神经机器翻译系统和大型语言模型。

**应用**：Transformer发展至今，最初用于构建执行特定任务的监督学习模型，后来作为自监督学习预训练模型的骨干网络，进一步引发了NLP范式的转变，取得了更显著的成功。目前主流的训练范式只需要在海量无标注文本上对模型进行预训练以学习语言的一般性知识，然后使用微调或提示等方法，模型即可轻松适应各类下游任务。近年来，由于Transformer可以模拟任何形式输入序列的强大能力，处理多模态数据变成现实，进一步成为跨模态的通用表示模型，应用场景十分广泛，涵盖了计算机视觉、语音处理和生物信息学等领域。

**大语言模型作为基础模型**：Transformer作为大语言模型（如GPT系列模型）的基础，在自然语言处理（NLP）乃至通用人工智能（AGI）领域展现出了惊人的突破。实际上，大语言模型的许多研究或多或少与Transformer相关，两者的发展进步相辅相成。一方面，训练大语言模型与训练Transformer解码器的联系十分紧密；对Transformer解码器的修改可以直接应用于大语言模型等等。另一方面，大语言模型的快速发展也推动了Transformer相关技术的进一步改进，例如将大型Transformer高效且低成本地适应于不同任务。

**理论分析**：尽管Transformer在各个领域中展现出了强大的能力，但与模型改进和工程化方面的广泛研究相比，其理论分析少之又少。这似乎不是Transformer的特有问题，而是自然语言处理和机器学习领域的共性问题。为此，研究人员正在努力从不同角度、不同工具挖掘Transformer的本质：1）通过数学工具解释Transformer中的深度神经网络，例如Transformer中的残差网络在数学上等同于常微分方程（ODE）的欧拉求解器，因此可以利用数值ODE方法的见解来指导模型设计；2）从理论层面解释自注意力机制，例如从机器学习如数据压缩、优化和函数近似的角度，解释自注意力和Transformer；3）借助形式系统解释Transformer，例如图灵机、计数器机器、正则语言和上下文无关语言、布尔电路、编程语言、一阶逻辑等。然而，值得关注的是，目前还没有学术界普遍认可的通用Transformer理论分析，可以帮助开发具有可解释和可预测行为的系统，这显然是机器学习领域复杂神经网络的一大挑战。


## Transformer相关论文列表

基于Transformer总览图中各个类别的关键模块，本项目补充整理了一份简短的Transformer相关论文列表，希望有助于NLP研究人员快速切入感兴趣的模块，辅助理解论文内容。注意，这不是一个详尽的论文列表，我们更希望这个列表可以作为快速了解Transformer基础和前沿内容的阅读资料。

#### [Transformer背景知识](#content)

1. **Neural Machine Translation by Jointly Learning to Align and Translate** ICLR 2015 [paper](https://arxiv.org/abs/1409.0473) *Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.*
2. **Sequence to Sequence Learning with Neural Networks** NeurIPS 2014 [paper](https://proceedings.neurips.cc/paper/2014/hash/a14ac55a4f27472c5d894ec1c3c743d2-Abstract.html) *Ilya Sutskever, Oriol Vinyals, Quoc V. Le.*
3. **Distributed Representations of Words and Phrases and their Compositionality** NeurIPS 2013 [paper](https://proceedings.neurips.cc/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html) *Tomas Mikolov, Ilya Sutskever, Kai Chen, Gregory S. Corrado, Jeffrey Dean.*
4. **A Neural Probabilistic Language Model** NeurIPS 2000 [paper](https://proceedings.neurips.cc/paper/2000/hash/728f206c2a01bf572b5940d7d9a8fa4c-Abstract.html) *Yoshua Bengio, Réjean Ducharme, Pascal Vincent.*
5. **Layer Normalization** NeurIPS 2016 [paper](https://openreview.net/forum?id=BJLa_ZC9) *Lei Jimmy Ba, Jamie Ryan Kiros, Geoffrey E. Hinton.*
6. **Deep Residual Learning for Image Recognition** CVPR 2016 [paper](https://ieeexplore.ieee.org/document/7780459/) *Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.*
7. **Improving Neural Networks by Preventing Co-Adaptation of Feature Detectors** arXiv 2012 [paper](https://arxiv.org/abs/1207.0580) *Geoffrey E. Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov.*
8. **Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation** arXiv 2016 [paper](https://arxiv.org/abs/1609.08144) *Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, Jeff Klingner, Apurva Shah, Melvin Johnson, Xiaobing Liu, Łukasz Kaiser, Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, Keith Stevens, George Kurian, Nishant Patil, Wei Wang, Cliff Young, Jason Smith, Jason Riesa, Alex Rudnick, Oriol Vinyals, Greg Corrado, Macduff Hughes, Jeffrey Dean.*
9. **Adam: A Method for Stochastic Optimization** ICLR 2015 [paper](https://arxiv.org/abs/1412.6980) *Diederik P. Kingma, Jimmy Ba.*


#### [位置编码](#content)

1. **Self-Attention with Relative Position Representations** NAACL 2018 [paper](https://aclanthology.org/N18-2074/) *Peter Shaw, Jakob Uszkoreit, Ashish Vaswani.*
2. **Transformer-XL: Attentive Language Models beyond a Fixed-Length Context** ACL 2019 [paper](https://aclanthology.org/P19-1285/) *Zihang Dai, Zhilin Yang, Yiming Yang, Jaime G. Carbonell, Quoc Viet Le, Ruslan Salakhutdinov.*
3. **DeBERTa: Decoding-Enhanced BERT with Disentangled Attention** ICLR 2021 [paper](https://openreview.net/forum?id=XPZIaotutsD) *Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen.*
4. **Roformer: Enhanced Transformer with Rotary Position Embedding** arXiv 2021 [paper](https://arxiv.org/abs/2104.09864) *Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu.*
5. **Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation** ICLR 2022 [paper](https://openreview.net/forum?id=R8sQPpGCv0) *Ofir Press, Noah Smith, Mike Lewis.*


#### [语法感知的注意力&探测](#content)

1. **What does BERT Look at? An Analysis of BERT's Attention** ACL 2019 [paper](https://aclanthology.org/W19-4828/) *Kevin Clark, Urvashi Khandelwal, Omer Levy, Christopher D. Manning.*
2. **Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned** ACL 2019 [paper](https://aclanthology.org/P19-1580/) *Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, Ivan Titov.*
3. **A Structural Probe for Finding Syntax in Word Representations** ACL 2019 [paper](https://aclanthology.org/N19-1419/) *John Hewitt, Christopher D. Manning.*
4. **What do You Learn from Context? Probing for Sentence Structure in Contextualized Word Representations** ICLR 2019 [paper](https://openreview.net/forum?id=SJzSgnRcKX) *Ian Tenney, Patrick Xia, Berlin Chen, Alex Wang, Adam Poliak, R. Thomas McCoy, Najoung Kim, Benjamin Van Durme, Samuel R. Bowman, Dipanjan Das, Ellie Pavlick.*
5. **Emergent Linguistic Structure in Artificial Neural Networks Trained by Self-Supervision** PNAS 2020 [paper](https://www.pnas.org/doi/full/10.1073/pnas.1907367117) *Christopher D. Manning, Kevin Clark, John Hewitt, Urvashi Khandelwal, Omer Levy.*


#### [稀疏注意力](#content)

1. **Longformer: The Long-Document Transformer** arXiv 2020 [paper](https://arxiv.org/abs/2004.05150) *Iz Beltagy, Matthew E. Peters, Arman Cohan.*
2. **Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting** AAAI 2021 [paper](https://ojs.aaai.org/index.php/AAAI/article/view/17325) *Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, Wancai Zhang.*
3. **Generating Long Sequences with Sparse Transformers** arXiv 2019 [paper](https://arxiv.org/abs/1904.10509) *Rewon Child, Scott Gray, Alec Radford, Ilya Sutskever.*
4. **Big bird: Transformers for Longer Sequences** NeurIPS 2020 [paper](https://proceedings.neurips.cc/paper/2020/hash/c8512d142a2d849725f31a9a7a361ab9-Abstract.html) *Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontañón, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed.*
5. **Image Transformer** ICML 2018 [paper](http://proceedings.mlr.press/v80/parmar18a.html) *Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Noam Shazeer, Alexander Ku, Dustin Tran.*
6. **Efficient Content-Based Sparse Attention with Routing Transformers** TACL 2021 [paper](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00353/97776/Efficient-Content-Based-Sparse-Attention-with) *Aurko Roy, Mohammad Saffar, Ashish Vaswani, David Grangier.*
7. **Adaptively Sparse Transformers** ACL 2019 [paper](https://aclanthology.org/D19-1223/) *Gonçalo M. Correia, Vlad Niculae, André F. T. Martins.*
8. **ETC: Encoding Long and Structured Inputs in Transformers** EMNLP 2020 [paper](https://aclanthology.org/2020.emnlp-main.19/) *Joshua Ainslie, Santiago Ontañón, Chris Alberti, Vaclav Cvicek, Zachary Fisher, Philip Pham, Anirudh Ravula, Sumit Sanghai, Qifan Wang, Li Yang.*
9. **Efficiently Modeling Long Sequences with Structured State Spaces** ICLR 2022 [paper](https://iclr.cc/virtual/2022/poster/6959) *Albert Gu, Karan Goel, Christopher Ré.*
10. **Sparse Sinkhorn Attention** ICML 2020 [paper](http://proceedings.mlr.press/v119/tay20a.html) *Yi Tay, Dara Bahri, Liu Yang, Donald Metzler, Da-Cheng Juan.*
11. **Adaptive Attention Span in Transformers** ACL 2019 [paper](https://aclanthology.org/P19-1032/) *Sainbayar Sukhbaatar, Edouard Grave, Piotr Bojanowski, Armand Joulin.*
12. **Efficient Transformers: A Survey** arXiv 2020 [paper](https://arxiv.org/abs/2009.06732) *Yi Tay, Mostafa Dehghani, Dara Bahri, Donald Metzler.*


#### [自注意力替代方法](#content)

1. **Pay Less Attention with Lightweight and Dynamic Convolutions** ICLR 2019 [paper](https://openreview.net/forum?id=SkVhlh09tX) *Felix Wu, Angela Fan, Alexei Baevski, Yann Dauphin, Michael Auli.*
2. **Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention** ICML 2020 [paper](https://proceedings.mlr.press/v119/katharopoulos20a.html) *Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, François Fleuret.*
3. **Rethinking Attention with Performers** ICLR 2021 [paper](https://openreview.net/forum?id=Ua6zuk0WRH&utm_campaign=NLP%20News&utm_medium=email&utm_source=Revue%20newsletter) *Krzysztof Marcin Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Quincy Davis, Afroz Mohiuddin, Lukasz Kaiser, David Benjamin Belanger, Lucy J Colwell, Adrian Weller.*
4. **Random Feature Attention** ICLR 2021 [paper](https://iclr.cc/virtual/2021/spotlight/3545) *Hao Peng, Nikolaos Pappas, Dani Yogatama, Roy Schwartz, Noah Smith, Lingpeng Kong.*
5. **Synthesizer: Rethinking Self-Attention for Transformer Models** ICML 2021 [paper](https://proceedings.mlr.press/v139/tay21a.html) *Yi Tay, Dara Bahri, Donald Metzler, Da-Cheng Juan, Zhe Zhao, Che Zheng.*
6. **On the Parameterization and Initialization of Diagonal State Space Models** NeurIPS 2022 [paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/e9a32fade47b906de908431991440f7c-Abstract-Conference.html) *Albert Gu, Karan Goel, Ankit Gupta, Christopher Ré.*
7. **Hungry Hungry Hippos: Towards Language Modeling with State Space Models** ICLR 2023 [paper](https://openreview.net/forum?id=COZDy0WYGg) *Daniel Y. Fu, Tri Dao, Khaled K. Saab, Armin W. Thomas, Atri Rudra, Christopher Ré.*


#### [结构改进](#content)

1. **Universal Transformers** ICLR 2019 [paper](https://openreview.net/forum?id=HyzdRiR9Y7) *Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, Lukasz Kaiser.*
2. **ODE Transformer: An Ordinary Differential Equation-Inspired Model for Sequence Generation** ACL 2022 [paper](https://aclanthology.org/2022.acl-long.571/) *Bei Li, Quan Du, Tao Zhou, Yi Jing, Shuhan Zhou, Xin Zeng, Tong Xiao, Jingbo Zhu, Xuebo Liu, Min Zhang.*
3. **Multiscale Vision Transformers** ICCV 2021 [paper](https://ieeexplore.ieee.org/document/9710800) *Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, Christoph Feichtenhofer.*
4. **Lite Transformer with Long-Short Range Attention** ICLR 2020 [paper](https://openreview.net/forum?id=ByeMPlHKPH) *Zhanghao Wu, Zhijian Liu, Ji Lin, Yujun Lin, Song Han.*
5. **The Evolved Transformer** ICML 2019 [paper](https://proceedings.mlr.press/v97/so19a) *David So, Quoc Le, Chen Liang.*


#### [深层模型](#content)

1. **Learning Deep Transformer Models for Machine Translation** ACL 2019 [paper](https://aclanthology.org/P19-1176/) *Qiang Wang, Bei Li, Tong Xiao, Jingbo Zhu, Changliang Li, Derek F. Wong, Lidia S. Chao.*
2. **Understanding the Difficulty of Training Transformers** EMNLP 2020 [paper](https://aclanthology.org/2020.emnlp-main.463/) *Liyuan Liu, Xiaodong Liu, Jianfeng Gao, Weizhu Chen, Jiawei Han.*
3. **Lipschitz Constrained Parameter Initialization for Deep Transformers** ACL 2020 [paper](https://aclanthology.org/2020.acl-main.38/) *Hongfei Xu, Qiuhui Liu, Josef van Genabith, Deyi Xiong, Jingyi Zhang.*
4. **Improving Deep Transformer with Depth-Scaled Initialization and Merged Attention** EMNLP 2019 [paper](https://aclanthology.org/D19-1083/) *Biao Zhang, Ivan Titov, Rico Sennrich.*
5. **Deepnet: Scaling Transformers to 1,000 Layers** arXiv 2022 [paper](https://arxiv.org/abs/2203.00555) *Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Dongdong Zhang, Furu Wei.*
6. **Learning Light-Weight Translation Models from Deep Transformer** AAAI 2021 [paper](https://ojs.aaai.org/index.php/AAAI/article/view/17561) *Bei Li, Ziyang Wang, Hui Liu, Quan Du, Tong Xiao, Chunliang Zhang, Jingbo Zhu.*


#### [宽度模型](#content)

1. **PaLM: Scaling Language Modeling with Pathways** JMLR 2023 [paper](https://jmlr.org/papers/v24/22-1144.html) *Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, Noah Fiedel.*
2. **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity** JMLR 2022 [paper](https://jmlr.org/papers/v23/21-0998.html) *William Fedus, Barret Zoph, Noam Shazeer.*
3. **GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding** ICLR 2021 [paper](https://openreview.net/forum?id=qrwe7XHTmYb) *Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, Zhifeng Chen.*
4. **Scaling Laws for Neural Language Models** arXiv 2020 [paper](https://arxiv.org/abs/2001.08361) *Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei.*


#### [循环&记忆&检索增强模型](#content)

1. **Compressive Transformers for Long-Range Sequence Modelling** ICLR 2020 [paper](https://openreview.net/forum?id=SylKikSYDH) *Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, Chloe Hillier, Timothy P. Lillicrap.*
2. **Accelerating Neural Transformer via an Average Attention Network** ACL 2018 [paper](https://aclanthology.org/P18-1166/) *Biao Zhang, Deyi Xiong, Jinsong Su.*
3. **The Best of Both Worlds: Combining Recent Advances in Neural Machine Translation** ACL 2018 [paper](https://aclanthology.org/P18-1008/) *Mia Xu Chen, Orhan Firat, Ankur Bapna, Melvin Johnson, Wolfgang Macherey, George Foster, Llion Jones, Mike Schuster, Noam Shazeer, Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Zhifeng Chen, Yonghui Wu, Macduff Hughes.*
4. **∞-Former: Infinite Memory Transformer** ACL 2022 [paper](https://aclanthology.org/2022.acl-long.375/) *Pedro Henrique Martins, Zita Marinho, André Martins.*
5. **Retrieval Augmented Language Model Pre-training** ICML 2020 [paper](https://proceedings.mlr.press/v119/guu20a.html?ref=https://githubhelp.com) *Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, Mingwei Chang.*
6. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** NeurIPS 2020 [paper](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html) *Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela.*
7. **Memorizing Transformers** ICLR 2022 [paper](https://iclr.cc/virtual/2022/spotlight/6065) *Yuhuai Wu, Markus Norman Rabe, DeLesley Hutchins, Christian Szegedy.*


#### [量化](#content)

1. **A White Paper on Neural Network Quantization** arXiv 2021 [paper](https://arxiv.org/abs/2106.08295) *Markus Nagel, Marios Fournarakis, Rana Ali Amjad, Yelysei Bondarenko, Mart van Baalen, Tijmen Blankevoort.*
2. **Training with Quantization Noise for Extreme Model Compression** ICLR 2021 [paper](https://openreview.net/forum?id=dV19Yyi1fS3) *Pierre Stock, Angela Fan, Benjamin Graham, Edouard Grave, Rémi Gribonval, Hervé Jégou, Armand Joulin.*
3. **Efficient 8-Bit Quantization of Transformer Neural Machine Language Translation Model** ICML 2019 [paper](https://arxiv.org/abs/1906.00532) *Aishwarya Bhandare, Vamsi Sripathi, Deepthi Karkada, Vivek Menon, Sun Choi, Kushal Datta, Vikram Saletore.*
4. **Fully Quantized Transformer for Machine Translation** EMNLP 2020 [paper](https://aclanthology.org/2020.findings-emnlp.1/) *Gabriele Prato, Ella Charlaix, Mehdi Rezagholizadeh.*
5. **Towards Fully 8-Bit Integer Inference for the Transformer Model** IJCAI 2020 [paper](https://dl.acm.org/doi/abs/10.5555/3491440.3491960) *Ye Lin, Yanyang Li, Tengbo Liu, Tong Xiao, Tongran Liu, Jingbo Zhu.*


#### [参数&激活共享](#content)

1. **Reformer: The Efficient Transformer** ICLR 2020 [paper](https://iclr.cc/virtual_2020/poster_rkgNKkHtvB.html) *Nikita Kitaev, Lukasz Kaiser, Anselm Levskaya.*
2. **Low Resource Dependency Parsing: Cross-lingual Parameter Sharing in a Neural Network Parser** ACL 2015 [paper](https://aclanthology.org/P15-2139/) *Long Duong, Trevor Cohn, Steven Bird, Paul Cook.*
3. **Sharing Attention Weights for Fast Transformer** IJCAI 2019 [paper](https://www.ijcai.org/Proceedings/2019/0735.pdf) *Tong Xiao, Yinqiao Li, Jingbo Zhu, Zhengtao Yu, Tongran Liu.*
4. **Fast Transformer Decoding: One Write-Head is All You Need** arXiv 2019 [paper](https://arxiv.org/pdf/1911.02150.pdf) *Noam Shazeer.*
5. **Parameter Sharing between Dependency Parsers for Related Languages** EMNLP 2018 [paper](https://arxiv.org/abs/1808.09055) *Miryam de Lhoneux, Johannes Bjerva, Isabelle Augenstein, Anders Søgaard.*


#### [压缩](#content)

1. **Sequence-Level Knowledge Distillation** EMNLP 2016 [paper](https://aclanthology.org/D16-1139/) *Yoon Kim, Alexander M. Rush.*
2. **Relational Knowledge Distillation** CVPR 2019 [paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html) *Wonpyo Park, Dongju Kim, Yan Lu, Minsu Cho.*
3. **Improved Knowledge Distillation via Teacher Assistant** AAAI 2020 [paper](https://ojs.aaai.org/index.php/AAAI/article/view/5963) *Seyed Iman Mirzadeh, Mehrdad Farajtabar, Ang Li, Nir Levine, Akihiro Matsukawa, Hassan Ghasemzadeh.*
4. **BPE-Dropout: Simple and Effective Subword Regularization** ACL 2020 [paper](https://aclanthology.org/2020.acl-main.170/) *Ivan Provilkov, Dmitrii Emelianenko, Elena Voita.*
5. **Block Pruning for Faster Transformers** EMNLP 2021 [paper](https://aclanthology.org/2021.emnlp-main.829/) *François Lagunas, Ella Charlaix, Victor Sanh, Alexander M. Rush.*
6. **Structured Pruning of Large Language Models** EMNLP 2020 [paper](https://aclanthology.org/2020.emnlp-main.496/) *Ziheng Wang, Jeremy Wohlwend, Tao Lei.*


#### [理论分析](#content)

1. **Theoretical Limitations of Self-Attention in Neural Sequence Models** TACL 2020 [paper](https://transacl.org/ojs/index.php/tacl/article/view/1815) *Michael Hahn.*
2. **Are Transformers Universal Approximators of Sequence-to-Sequence Functions?** ICLR 2020 [paper](https://openreview.net/forum?id=ByxRM0Ntvr) *Chulhee Yun, Srinadh Bhojanapalli, Ankit Singh Rawat, Sashank Reddi, Sanjiv Kumar.*
3. **On the Turing Completeness of Modern Neural Network Architectures** ICLR 2019 [paper](https://openreview.net/forum?id=HyGBdo0qFm) *Jorge Pérez, Javier Marinkovic, Pablo Barceló.*
4. **A Theoretical Understanding of Shallow Vision Transformers: Learning, Generalization, and Sample Complexity** ICLR 2023 [paper](https://openreview.net/pdf?id=jClGv3Qjhb) *Hongkang Li, Meng Wang, Sijia Liu, Pin-Yu Chen.*
5. **Saturated Transformers are Constant-Depth Threshold Circuits** TACL 2022 [paper](https://transacl.org/index.php/tacl/article/view/3465) *William Merrill, Ashish Sabharwal, Noah A. Smith.*
6. **Transformers as Recognizers of Formal Languages: A Survey on Expressivity** arXiv 2023 [paper](https://arxiv.org/abs/2311.00208) *Lena Strobl, William Merrill, Gail Weiss, David Chiang, Dana Angluin.*
7. **Low-Rank Bottleneck in Multi-head Attention Models** ICML 2020 [paper](https://proceedings.mlr.press/v119/bhojanapalli20a.html) *Srinadh Bhojanapalli, Chulhee Yun, Ankit Singh Rawat, Sashank Reddi, Sanjiv Kumar.*


#### [用于语言理解的预训练Transformer](#content)

1. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** NAACL 2019 [paper](https://aclanthology.org/N19-1423/) *Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova.*
2. **SpanBERT: Improving Pre-training by Representing and Predicting Spans** TACL 2020 [paper](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00300/43539/SpanBERT-Improving-Pre-training-by-Representing) *Mandar Joshi, Danqi Chen, Yinhan Liu, Daniel S. Weld, Luke Zettlemoyer, Omer Levy.*
3. **ALBERT: A Lite BERT for Self-Supervised Learning of Language Representations** ICLR 2020 [paper](https://openreview.net/forum?id=H1eA7AEtvS) *Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut.*
4. **RoBERTa: A Robustly Optimized BERT Pretraining Approach** arXiv 2019 [paper](https://arxiv.org/abs/1907.11692) *Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov.*
5. **XLNet: Generalized Autoregressive Pretraining for Language Understanding** NeurIPS 2019 [paper](https://proceedings.neurips.cc/paper_files/paper/2019/hash/dc6a7e655d7e5840e66733e9ee67cc69-Abstract.html) *Zhilin Yang, Zihang Dai, Yiming Yang, Jaime G. Carbonell, Ruslan Salakhutdinov, Quoc V. Le.*
6. **Unsupervised Cross-Lingual Representation Learning at Scale** ACL 2020 [paper](https://aclanthology.org/2020.acl-main.747/) *Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, Veselin Stoyanov.*


#### [用于语言生成的预训练Transformer](#content)

1. **Language Models are Few-Shot Learners** NeurIPS 2020 [paper](https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html) *Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei.*
2. **LLaMA: Open and Efficient Foundation Language Models** arXiv 2023 [paper](https://arxiv.org/abs/2302.13971) *Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample.*
3. **BLOOM: A 176B-Parameter Open-Access Multilingual Language Model** arXiv 2022 [paper](https://arxiv.org/abs/2211.05100) *Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilic, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, Jonathan Tow, Alexander M. Rush, Stella Biderman, Albert Webson, Pawan Sasanka Ammanamanchi, Thomas Wang, Benoît Sagot, Niklas Muennighoff, Albert Villanova del Moral, Olatunji Ruwase, Rachel Bawden, Stas Bekman, Angelina McMillan-Major, Iz Beltagy, Huu Nguyen, Lucile Saulnier, Samson Tan, Pedro Ortiz Suarez, Victor Sanh, Hugo Laurençon, Yacine Jernite, Julien Launay, Margaret Mitchell, Colin Raffel, Aaron Gokaslan, Adi Simhi, Aitor Soroa, Alham Fikri Aji, Amit Alfassy, Anna Rogers, Ariel Kreisberg Nitzav, Canwen Xu, Chenghao Mou, Chris Emezue, Christopher Klamm, Colin Leong, Daniel van Strien, David Ifeoluwa Adelani, et al..*
4. **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer** JMLR 2020 [paper](https://dl.acm.org/doi/abs/10.5555/3455716.3455856) *Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu.*
5. **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism** arXiv 2019 [paper](https://arxiv.org/abs/1909.08053) *Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro.*
6. **MASS: Masked Sequence to Sequence Pre-training for Language Generation** ICML 2019 [paper](http://proceedings.mlr.press/v97/song19d.html) *Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, Tie-Yan Liu.*
7. **BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension** ACL 2020 [paper](https://aclanthology.org/2020.acl-main.703/?utm_campaign=The+Batch&%3Butm_source=hs_email&%3Butm_medium=email&%3B_hsenc=p2ANqtz-812IhL294q5bT5M5HLvLxD6pL7M9lE2Hd0-wf5UNphYYcVx-f2K7KwaNh68AO8zDpN8Vfv&ref=dl-staging-website.ghost.io) *Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, Luke Zettlemoyer.*


#### [其他应用](#content)

1. **Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows** ICCV 2021 [paper](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper) *Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo.*
2. **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale** ICLR 2021 [paper](https://openreview.net/forum?id=YicbFdNTTy) *Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby.*
3. **Conformer: Convolution-Augmented Transformer for Speech Recognition** INTERSPEECH 2020 [paper](https://www.isca-speech.org/archive/interspeech_2020/gulati20_interspeech.html) *Anmol Gulati, James Qin, Chung-Cheng Chiu, Niki Parmar, Yu Zhang, Jiahui Yu, Wei Han, Shibo Wang, Zhengdong Zhang, Yonghui Wu, Ruoming Pang.*
4. **Wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations** NeurIPS 2020 [paper](https://proceedings.neurips.cc/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html) *Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, Michael Auli.*
5. **Data2vec: A General Framework for Self-Supervised Learning in Speech, Vision and Language** ICML 2022 [paper](https://proceedings.mlr.press/v162/baevski22a.html) *Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, Michael Auli.*
6. **ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks** NeurIPS 2019 [paper](https://papers.nips.cc/paper/2019/hash/c74d97b01eae257e44aa9d5bade97baf-Abstract.html) *Jiasen Lu, Dhruv Batra, Devi Parikh, Stefan Lee.*


## （可能）有用的资源

1. [系统] [Fairseq](https://github.com/facebookresearch/fairseq)
2. [系统] [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor)
3. [系统] [Huggingface Transformers](https://github.com/huggingface/transformers)
4. [系统] [FasterTransformer](https://github.com/NVIDIA/FasterTransformer)
5. [系统] [BERT](https://github.com/google-research/bert)
6. [系统] [LLaMA](https://github.com/facebookresearch/llama)
7. [教程] [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
8. [教程] [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
9. [教程] [Attention Mechanisms and Transformers (Dive into Deep Learning)](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html)
10. [教程] [Transformers and Multi-Head Attention (UvA Deep Learning Tutorials)](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html)


## 致谢

感谢东北大学自然语言处理实验室穆永誉、王成龙、李北、单韦乔、范瑀纯、常开妍、郑童、鲍慧雯为这项工作所做的贡献。

如有任何问题和意见，请发送电子邮件至 xiaotong [at] mail.neu.edu.cn 或者 heshengmo [at] foxmail.com。
