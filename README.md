# Introduction to Transformers: an NLP Perspective

Transformers have dominated empirical machine learning models of natural language processing (NLP). Here we introduce basic concepts of Transformers and present key techniques that form the recent advances of these models. This includes a description of the standard Transformer architecture, a series of model refinements, and common applications. Given that Transformers and related deep learning techniques might be evolving in ways we have never seen, we cannot dive into all the model details or cover all the technical areas. Instead, we focus on just those concepts that are helpful for gaining a good understanding of Transformers and their variants. We also summarize the key ideas that impact this field, thereby yielding some insights into the strengths and limitations of these models.

You can find the pdf file here: [Introduction to Transformers: an NLP Perspective](https://www.neu.edu.cn/). 

This work is intended for students and researchers in NLP who have basic knowledge of linear algebra and probabilities. Although some familiarity with machine learning (in particular, neural networks and deep learning) is advantageous, readers can still gain a general understanding of Transformers by skipping the sections or sub-sections that require specialized background knowledge.

## Selected Papers

For reference, we select some papers for each of the topics. There is such a vast amount of research that it is impossible to provide a complete list of related works. Instead of attempting a comprehensive survey of all research areas, we provide a very short list of papers to facilitate a quick understanding of the key issues. 

![transformer-pic](figures/figure-transformer-roadmap.jpg#pic_center)

#### [Background Knowledge for Learning Transformers](#content)


1. **Neural Machine Translation by Jointly Learning to Align and Translate** ICLR 2015 [paper](https://arxiv.org/abs/1409.0473) *Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.*
2. **Sequence to Sequence Learning with Neural Networks** NeurlPS 2014 [paper](https://proceedings.neurips.cc/paper/2014/hash/a14ac55a4f27472c5d894ec1c3c743d2-Abstract.html) *Ilya Sutskever, Oriol Vinyals, Quoc V. Le.*
3. **Distributed Representations of Words and Phrases and their Compositionality** NeurlPS 2013 [paper](https://proceedings.neurips.cc/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html) *Tomás Mikolov, Ilya Sutskever, Kai Chen, Gregory S. Corrado, Jeffrey Dean.*
4. **A Neural Probabilistic Language Model** NeurlPS 2000 [paper](https://proceedings.neurips.cc/paper/2000/hash/728f206c2a01bf572b5940d7d9a8fa4c-Abstract.html) *Yoshua Bengio, Réjean Ducharme, Pascal Vincent.*
5. **Layer Normalization** NeurlPS 2016 [paper](https://openreview.net/forum?id=BJLa_ZC9) *Lei Jimmy Ba, Jamie Ryan Kiros, Geoffrey E. Hinton.*
6. **Deep Residual Learning for Image Recognition** CVPR 2016 [paper](https://ieeexplore.ieee.org/document/7780459/) *Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.*
7. **Improving Neural Networks by Preventing Co-Adaptation of Feature Detectors** arXiv 2012 [paper](https://arxiv.org/abs/1207.0580) *Geoffrey E. Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov.*
8. **Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation** arXiv 2016 [paper](https://arxiv.org/abs/1609.08144) *Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, Jeff Klingner, Apurva Shah, Melvin Johnson, Xiaobing Liu, Łukasz Kaiser, Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, Keith Stevens, George Kurian, Nishant Patil, Wei Wang, Cliff Young, Jason Smith, Jason Riesa, Alex Rudnick, Oriol Vinyals, Greg Corrado, Macduff Hughes, Jeffrey Dean*
9. **Adam: A Method for Stochastic Optimization** ICLR 2015 [paper](https://arxiv.org/abs/1412.6980) *Diederik P. Kingma, Jimmy Ba*

#### [Positional Encoding](#content)

1. **Self-Attention with Relative Position Representations** NAACL 2018 [paper](https://aclanthology.org/N18-2074/) *Peter Shaw, Jakob Uszkoreit, Ashish Vaswani.*
2. **Transformer-XL: Attentive Language Models beyond a Fixed-Length Context** ACL 2019 [paper](https://aclanthology.org/P19-1285/) *Zihang Dai, Zhilin Yang, Yiming Yang, Jaime G. Carbonell, Quoc Viet Le, Ruslan Salakhutdinov.*
3. **DeBERTa: Decoding-Enhanced BERT with Disentangled Attention** ICLR 2021 [paper](https://openreview.net/forum?id=XPZIaotutsD) *Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen.*
4. **Roformer: Enhanced Transformer with Rotary Position Embedding** arXiv 2021 [paper](https://arxiv.org/abs/2104.09864) *Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, Yunfeng Liu.*
5. **Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation** ICLR 2022 [paper](https://openreview.net/forum?id=R8sQPpGCv0) *Ofir Press, Noah A. Smith, Mike Lewis.*


#### [Syntax-aware Attention & Probing](#content)

1. **What does BERT Look at? An Analysis of BERT's Attention** ACL 2019 [paper](https://aclanthology.org/W19-4828/) *Kevin Clark, Urvashi Khandelwal, Omer Levy, Christopher D. Manning.*
2. **Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned** ACL 2019 [paper](https://aclanthology.org/P19-1580/) *Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, Ivan Titov.*
3. **A Structural Probe for Finding Syntax in Word Representations** ACL 2019 [paper](https://aclanthology.org/N19-1419/) *John Hewitt, Christopher D. Manning.*
4. **What do You Learn from Context? Probing for Sentence Structure in Contextualized Word Representations** ICLR 2019 [paper](https://openreview.net/forum?id=SJzSgnRcKX) *Ian Tenney, Patrick Xia, Berlin Chen, Alex Wang, Adam Poliak, R. Thomas McCoy, Najoung Kim, Benjamin Van Durme, Samuel R. Bowman, Dipanjan Das, Ellie Pavlick.*
5. **Emergent Linguistic Structure in Artificial Neural Networks Trained by Self-Supervision** NAS 2020 [paper](https://www.pnas.org/doi/full/10.1073/pnas.1907367117) *Christopher D. Manning, Kevin Clark, John Hewitt, Urvashi Khandelwal, Omer Levy.*


#### [Sparse Attention](#content)

1. **Longformer: The Long-Document Transformer** arXiv 2020 [paper](https://arxiv.org/abs/2004.05150) *Iz Beltagy, Matthew E. Peters, Arman Cohan.*
2. **Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting** AAAI 2021 [paper](https://ojs.aaai.org/index.php/AAAI/article/view/17325) *Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, Wancai Zhang.*
3. **Generating Long Sequences with Sparse Transformers** arXiv 2019 [paper](https://arxiv.org/abs/1904.10509) *Rewon Child, Scott Gray, Alec Radford, Ilya Sutskever.*
4. **Big bird: Transformers for Longer Sequences** NeurlPS 2020 [paper](https://proceedings.neurips.cc/paper/2020/hash/c8512d142a2d849725f31a9a7a361ab9-Abstract.html) *Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontañón, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed.*
5. **Image Transformer** ICML 2018 [paper](http://proceedings.mlr.press/v80/parmar18a.html) *Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Noam Shazeer, Alexander Ku, Dustin Tran.*
6. **Efficient Content-Based Sparse Attention with Routing Transformers** TACL 2021 [paper](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00353/97776/Efficient-Content-Based-Sparse-Attention-with) *Aurko Roy, Mohammad Saffar, Ashish Vaswani, David Grangier.*
7. **Adaptively Sparse Transformers** ACL 2019 [paper](https://aclanthology.org/D19-1223/) *Gonçalo M. Correia, Vlad Niculae, André F. T. Martins.*
8. **ETC: Encoding Long and Structured Inputs in Transformers** EMNLP 2020 [paper](https://aclanthology.org/2020.emnlp-main.19/) *Joshua Ainslie, Santiago Ontañón, Chris Alberti, Vaclav Cvicek, Zachary Fisher, Philip Pham, Anirudh Ravula, Sumit Sanghai, Qifan Wang, Li Yang.*
9. **Efficiently Modeling Long Sequences with Structured State Spaces** ICLR 2022 [paper](https://iclr.cc/virtual/2022/poster/6959) *Albert Gu, Karan Goel, Christopher Ré.*
10. **Lite Transformer with Long-Short Range Attention** ICLR 2020 [paper](https://openreview.net/forum?id=ByeMPlHKPH) *Zhanghao Wu, Zhijian Liu, Ji Lin, Yujun Lin, Song Han.*
11. **Sparse Sinkhorn Attention** ICML 2020 [paper](http://proceedings.mlr.press/v119/tay20a.html) *Yi Tay, Dara Bahri, Liu Yang, Donald Metzler, Da-Cheng Juan.*
12. **Adaptive Attention Span in Transformers** ACL 2019 [paper](https://aclanthology.org/P19-1032/) *Sainbayar Sukhbaatar, Edouard Grave, Piotr Bojanowski, Armand Joulin.*
13. **Efficient Transformers: A Survey** arXiv 2020 [paper](https://arxiv.org/abs/2009.06732) *Yi Tay, Mostafa Dehghani, Dara Bahri, Donald Metzler.*

#### [Alternatives to Self-Attention](#content)

1. **Pay Less Attention with Lightweight and Dynamic Convolutions** ICLR 2019 [paper](https://openreview.net/forum?id=SkVhlh09tX) *Felix Wu, Angela Fan, Alexei Baevski, Yann Dauphin, Michael Auli.*
2. **Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention** ICML 2020 [paper](https://proceedings.mlr.press/v119/katharopoulos20a.html) *Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, François Fleuret.*
3. **Rethinking Attention with Performers** ICLR 2021 [paper](https://openreview.net/forum?id=Ua6zuk0WRH&utm_campaign=NLP%20News&utm_medium=email&utm_source=Revue%20newsletter) *Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, David Belanger, Lucy Colwell, Adrian Weller.*
4. **Random Feature Attention** ICLR 2021 [paper](https://iclr.cc/virtual/2021/spotlight/3545) *Hao Peng, Nikolaos Pappas, Dani Yogatama, Roy Schwartz, Noah A. Smith, Lingpeng Kong.*
5. **Synthesizer: Rethinking Self-Attention in Transformer Models** ICML 2021 [paper](https://proceedings.mlr.press/v139/tay21a.html) *Yi Tay, Dara Bahri, Donald Metzler, Da-Cheng Juan, Zhe Zhao, Che Zheng.*
6. **On the Parameterization and Initialization of Diagonal State Space Models** NeurIPS 2022 [paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/e9a32fade47b906de908431991440f7c-Abstract-Conference.html) *Albert Gu, Ankit Gupta, Karan Goel, Christopher Ré.*
7. **Hungry Hungry Hippos: Towards Language Modeling with State Space Models** ICLR 2023 [paper](https://openreview.net/forum?id=COZDy0WYGg) *Daniel Y. Fu, Tri Dao, Khaled K. Saab, Armin W. Thomas, Atri Rudra, Christopher Ré.*


#### [Architecture Improvement](#content)

1. **Universal Transformers** ICLR 2019 [paper](https://openreview.net/forum?id=HyzdRiR9Y7) *Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, Lukasz Kaiser.*
2. **ODE Transformer: An Ordinary Differential Equation-Inspired Model for Sequence Generation** ACL 2022 [paper](https://aclanthology.org/2022.acl-long.571/) *Bei Li, Quan Du, Tao Zhou, Yi Jing, Shuhan Zhou, Xin Zeng, Tong Xiao, Jingbo Zhu, Xuebo Liu, Min Zhang.*
3. **Multiscale Vision Transformers** ICCV 2021 [paper](https://ieeexplore.ieee.org/document/9710800) *Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, Christoph Feichtenhofer.*
4. **Lite Transformer with Long-Short Range Attention** ICLR 2020 [paper](https://openreview.net/forum?id=ByeMPlHKPH) *Zhanghao Wu, Zhijian Liu, Ji Lin, Yujun Lin, Song Han.*
5. **The Evolved Transformer** ICML 2019 [paper](https://proceedings.mlr.press/v97/so19a) *David R. So, Chen Liang, Quoc V. Le.*


#### [Deep Models](#content)

1. **Learning Deep Transformer Models for Machine Translation** ACL 2019 [paper](https://aclanthology.org/P19-1176/) *Qiang Wang, Bei Li, Tong Xiao, Jingbo Zhu, Changliang Li, Derek F. Wong, Lidia S. Chao.*
2. **Understanding the Difficulty of Training Transformers** EMNLP 2020 [paper](https://aclanthology.org/2020.emnlp-main.463/) *Liyuan Liu, Xiaodong Liu, Jianfeng Gao, Weizhu Chen, Jiawei Han.*
3. **Lipschitz Constrained Parameter Initialization for Deep Transformers** ACL 2020 [paper](https://aclanthology.org/2020.acl-main.38/) *Hongfei Xu, Qiuhui Liu, Josef van Genabith, Deyi Xiong, Jingyi Zhang.*
4. **Improving Deep Transformer with Depth-Scaled Initialization and Merged Attention** EMNLP 2019 [paper](https://aclanthology.org/D19-1083/) *Biao Zhang, Ivan Titov, Rico Sennrich.*
5. **Deepnet: Scaling Transformers to 1,000 Layers** arXiv 2022 [paper](https://arxiv.org/abs/2203.00555) *Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Dongdong Zhang, Furu Wei.*
6. **Learning Light-Weight Translation Models from Deep Transformer** AAAI 2021 [paper](https://ojs.aaai.org/index.php/AAAI/article/view/17561) *Bei Li, Ziyang Wang, Hui Liu, Quan Du, Tong Xiao, Chunliang Zhang, Jingbo Zhu.*


#### [Wide Models](#content)

1. **Palm: Scaling Language Modeling with Pathways** JMLR 2023 [paper](https://jmlr.org/papers/v24/22-1144.html) *Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, Noah Fiedel.*
2. **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity** JMLR 2022 [paper](https://jmlr.org/papers/v23/21-0998.html) *William Fedus, Barret Zoph, Noam Shazeer.*
3. **Gshard: Scaling Giant Models with Conditional Computation and Automatic Sharding** ICLR 2021 [paper](https://openreview.net/forum?id=qrwe7XHTmYb) *Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, Zhifeng Chen.*
4. **Scaling Laws for Neural Language Models** arXiv 2020 [paper](https://arxiv.org/abs/2001.08361) *Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei.*


#### [Recurrent & Memory & Retrieval-Augmented Models](#content)

1. **Compressive Transformers for Long-Range Sequence Modelling** ICLR 2020 [paper](https://openreview.net/forum?id=SylKikSYDH) *Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, Chloe Hillier, Timothy P. Lillicrap.*
2. **Accelerating Neural Transformer via an Average Attention Network** ACL 2018 [paper](https://aclanthology.org/P18-1166/) *Biao Zhang, Deyi Xiong, Jinsong Su.*
3. **The Best of Both Worlds: Combining Recent Advances in Neural Machine Translation** ACL 2018 [paper](https://aclanthology.org/P18-1008/) *Mia Xu Chen, Orhan Firat, Ankur Bapna, Melvin Johnson, Wolfgang Macherey, George Foster, Llion Jones, Niki Parmar, Mike Schuster, Zhifeng Chen, Yonghui Wu, Macduff Hughes.*
4. **∞-Former: Infinite Memory Transformer** ACL 2022 [paper](https://aclanthology.org/2022.acl-long.375/) *Pedro Henrique Martins, Zita Marinho, André F. T. Martins.*
5. **Retrieval Augmented Language Model Pre-training** ICML 2020 [paper](https://proceedings.mlr.press/v119/guu20a.html?ref=https://githubhelp.com) *Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, Ming-Wei Chang.*
6. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** NeurIPS 2020 [paper](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html) *Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela.*
7. **Memorizing Transformers** ICLR 2022 [paper](https://iclr.cc/virtual/2022/spotlight/6065) *Yuhuai Wu, Markus Norman Rabe, DeLesley Hutchins, Christian Szegedy.*


#### [Quantization](#content)

1. **A White Paper on Neural Network Quantization** arXiv 2021 [paper](https://arxiv.org/abs/2106.08295) *Markus Nagel, Marios Fournarakis, Rana Ali Amjad, Yelysei Bondarenko, Mart van Baalen, Tijmen Blankevoort.*
2. **Training with Quantization Noise for Extreme Model Compression** ICLR 2021 [paper](https://openreview.net/forum?id=dV19Yyi1fS3) *Pierre Stock, Angela Fan, Benjamin Graham, Edouard Grave, Rémi Gribonval, Hervé Jégou, Armand Joulin.*
3. **Efficient 8-Bit Quantization of Transformer Neural Machine Language Translation Model** ICML 2019 [paper](https://arxiv.org/abs/1906.00532) *Aishwarya Bhandare, Vamsi Sripathi, Deepthi Karkada, Vivek Menon, Sun Choi, Kushal Datta, Vikram A. Saletore.*
4. **Fully Quantized Transformer for Machine Translation** EMNLP 2020 [paper](https://aclanthology.org/2020.findings-emnlp.1/) *Gabriele Prato, Ella Charlaix, Mehdi Rezagholizadeh.*
5. **Towards Fully 8-Bit Integer Inference for the Transformer Model** IJCAI 2020 [paper](https://dl.acm.org/doi/abs/10.5555/3491440.3491960) *Ye Lin, Yanyang Li, Tengbo Liu, Tong Xiao, Tongran Liu, Jingbo Zhu.*

#### [Parameter & Activation Sharing](#content)

1. **Reformer: The Efficient Transformer** ICLR 2020 [paper](https://iclr.cc/virtual_2020/poster_rkgNKkHtvB.html) *Nikita Kitaev, Lukasz Kaiser, Anselm Levskaya.*
2. **Low Resource Dependency Parsing: Cross-lingual Parameter Sharing in a Neural Network Parser** ACL 2015 [paper](https://aclanthology.org/P15-2139/) *Long Duong, Trevor Cohn, Steven Bird, Paul Cook.*
3. **Sharing Attention Weights for Fast Transformer** IJCAI 2019 [paper](https://www.ijcai.org/Proceedings/2019/0735.pdf) *Tong Xiao, Yinqiao Li, Jingbo Zhu, Zhengtao Yu, Tongran Liu.*
4. **Fast Transformer Decoding: One Write-Head is All You Need** arXiv 2019 [paper](https://arxiv.org/pdf/1911.02150.pdf) *Noam Shazeer.*
5. **Parameter Sharing between Dependency Parsers for Related Languages** EMNLP 2018 [paper](https://arxiv.org/abs/1808.09055) *Miryam de Lhoneux, Johannes Bjerva, Isabelle Augenstein, Anders Søgaard.*

#### [Compression](#content)

1. **Sequence-Level Knowledge Distillation** EMNLP 2016 [paper](https://aclanthology.org/D16-1139/) *Yoon Kim, Alexander M. Rush.*
2. **Relational Knowledge Distillation** CVPR 2019 [paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html) *Wonpyo Park, Dongju Kim, Yan Lu, Minsu Cho.*
3. **Improved Knowledge Distillation via Teacher Assistant** AAAI 20 [paper](https://ojs.aaai.org/index.php/AAAI/article/view/5963) *Seyed-Iman Mirzadeh, Mehrdad Farajtabar, Ang Li, Nir Levine, Akihiro Matsukawa, Hassan Ghasemzadeh.*
4. **BPE-Dropout: Simple and Effective Subword Regularization** ACL 2020 [paper](https://aclanthology.org/2020.acl-main.170/) *Ivan Provilkov, Dmitrii Emelianenko, Elena Voita.*
5. **Block Pruning for Faster Transformers** EMNLP 2021 [paper](https://aclanthology.org/2021.emnlp-main.829/) *François Lagunas, Ella Charlaix, Victor Sanh, Alexander M. Rush.*
6. **Structured Pruning of Large Language Models** EMNLP 2020 [paper](https://aclanthology.org/2020.emnlp-main.496/) *Ziheng Wang, Jeremy Wohlwend, Tao Lei.*


#### [Theoretical Analysis](#content)

1. **Theoretical Limitations of Self-Attention in Neural Sequence Models** TACL 2020 [paper](https://transacl.org/ojs/index.php/tacl/article/view/1815) *Michael Hahn.*
2. **Are Transformers Universal Approximators of Sequence-to-Sequence Functions?** ICLR 2020 [paper](https://openreview.net/forum?id=ByxRM0Ntvr) *Chulhee Yun, Srinadh Bhojanapalli, Ankit Singh Rawat, Sashank J. Reddi, Sanjiv Kumar.*
3. **On the Turing Completeness of Modern Neural Network Architectures** ICLR 2019 [paper](https://openreview.net/forum?id=HyGBdo0qFm) *Jorge Pérez, Javier Marinkovic, Pablo Barceló.*
4. **A Theoretical Understanding of Shallow Vision Transformers: Learning, Generalization, and Sample Complexity** ICLR 2023 [paper](https://openreview.net/pdf?id=jClGv3Qjhb) *Hongkang Li, Meng Wang, Sijia Liu, Pin-Yu Chen.*
5. **Saturated Transformers are Constantdepth Threshold Circuits** TACL 2022 [paper](https://transacl.org/index.php/tacl/article/view/3465) *William Merrill, Ashish Sabharwal, Noah A. Smith.*
6. **Transformers as Recognizers of Formal Languages: A Survey on Expressivity** arXiv 2023 [paper](https://arxiv.org/abs/2311.00208) *Lena Strobl, William Merrill, Gail Weiss, David Chiang, Dana Angluin.*
7. **Low-Rank Bottleneck in Multi-head Attention Models** ICML 2020 [paper](https://proceedings.mlr.press/v119/bhojanapalli20a.html) *Srinadh Bhojanapalli, Chulhee Yun, Ankit Singh Rawat, Sashank J. Reddi, Sanjiv Kumar.*


#### [Pre-trained Transformers for Language Understanding](#content)

1. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** NAACL 2019 [paper](https://aclanthology.org/N19-1423/) *Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova.*
2. **SpanBERT: Improving Pre-training by Representing and Predicting Spans** TACL 2020 [paper](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00300/43539/SpanBERT-Improving-Pre-training-by-Representing) *Mandar Joshi, Danqi Chen, Yinhan Liu, Daniel S. Weld, Luke Zettlemoyer, Omer Levy.*
3. **ALBERT: A Lite BERT for Self-Supervised Learning of Language Representations** ICLR 2020 [paper](https://openreview.net/forum?id=H1eA7AEtvS) *Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut.*
4. **RoBERTa: A Robustly Optimized BERT Pretraining Approach** arXiv 2019 [paper](https://arxiv.org/abs/1907.11692) *Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov.*
5. **XLNet: Generalized Autoregressive Pretraining for Language Understanding** NeurIPS 2019 [paper](https://arxiv.org/abs/1906.08237) *Zhilin Yang, Zihang Dai, Yiming Yang, Jaime G. Carbonell, Ruslan Salakhutdinov, Quoc V. Le.*
6. **Unsupervised Cross-Lingual Representation Learning at Scale** ACL 2020 [paper](https://aclanthology.org/2020.acl-main.747/) *Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, Veselin Stoyanov.*

#### [Pre-trained Transformers for Language Generation](#content)

1. **Language Models are Few-Shot Learners** NeurlPS 2020 [paper](https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html) *Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei.*
2. **LLaMA: Open and Efficient Foundation Language Models** arXiv 2023 [paper](https://arxiv.org/abs/2302.13971) *Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample.*
3. **BLOOM: A 176B-Parameter Open-Access Multilingual Language Model** arXiv 2022 [paper](https://arxiv.org/abs/2211.05100) *Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilic, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, Jonathan Tow, Alexander M. Rush, Stella Biderman, Albert Webson, Pawan Sasanka Ammanamanchi, Thomas Wang, Benoît Sagot, Niklas Muennighoff, Albert Villanova del Moral, Olatunji Ruwase, Rachel Bawden, Stas Bekman, Angelina McMillan-Major, Iz Beltagy, Huu Nguyen, Lucile Saulnier, Samson Tan, Pedro Ortiz Suarez, Victor Sanh, Hugo Laurençon, Yacine Jernite, Julien Launay, Margaret Mitchell, Colin Raffel, Aaron Gokaslan, Adi Simhi, Aitor Soroa, Alham Fikri Aji, Amit Alfassy, Anna Rogers, Ariel Kreisberg Nitzav, Canwen Xu, Chenghao Mou, Chris Emezue, Christopher Klamm, Colin Leong, Daniel van Strien, David Ifeoluwa Adelani, et al..*
4. **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer** JMLR 2020 [paper](https://dl.acm.org/doi/abs/10.5555/3455716.3455856) *Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu.*
5. **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism** arXiv 2019 [paper](https://arxiv.org/abs/1909.08053) *Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro.*
6. **MASS: Masked Sequence to Sequence Pre-training for Language Generation** ICML 2019 [paper](http://proceedings.mlr.press/v97/song19d.html) *Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, Tie-Yan Liu.*
7. **BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension** ACL 2020 [paper](https://aclanthology.org/2020.acl-main.703/?utm_campaign=The+Batch&%3Butm_source=hs_email&%3Butm_medium=email&%3B_hsenc=p2ANqtz-812IhL294q5bT5M5HLvLxD6pL7M9lE2Hd0-wf5UNphYYcVx-f2K7KwaNh68AO8zDpN8Vfv&ref=dl-staging-website.ghost.io) *Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, Luke Zettlemoyer.*

#### [Other Applications](#content)

1. **Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows** ICCV 2021 [paper](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper) *Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo.*
2. **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale** ICLR 2021 [paper](https://openreview.net/forum?id=YicbFdNTTy) *Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby.*
3. **Conformer: Convolution-Augmented Transformer for Speech Recognition** INTERSPEECH 2020 [paper](https://www.isca-speech.org/archive/interspeech_2020/gulati20_interspeech.html) *Anmol Gulati, James Qin, Chung-Cheng Chiu, Niki Parmar, Yu Zhang, Jiahui Yu, Wei Han, Shibo Wang, Zhengdong Zhang, Yonghui Wu, Ruoming Pang.*
4. **Wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations** NeurIPS 2020 [paper](https://proceedings.neurips.cc/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html) *Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, Michael Auli.*
5. **Data2vec: A General Framework for Self-Supervised Learning in Speech, Vision and Language** ICML 2022 [paper](https://proceedings.mlr.press/v162/baevski22a.html) *Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, Michael Auli.*
6. **ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks** NeurIPS 2019 [paper](https://arxiv.org/abs/1908.02265) *Jiasen Lu, Dhruv Batra, Devi Parikh, Stefan Lee.*

## Useful Resources

1. [Fairseq](https://github.com/facebookresearch/fairseq)
2. [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor)
3. [Huggingface Transformers](https://github.com/huggingface/transformers)
4. [FasterTransformer](https://github.com/NVIDIA/FasterTransformer)
5. [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
6. [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
7. [Attention Mechanisms and Transformers (Dive into Deep Learning)](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html)
8. [Transformers and Multi-Head Attention (UvA Deep Learning Tutorials)](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html)

## Acknowledgements

We would like to thank Yongyu Mu, Chenglong Wang, Bei Li, Weiqiao Shan, Yuchun Fan, Kaiyan Chang, Tong Zheng, and Huiwen Bao for their contributions to this work.

For any questions and comments, please email us at xiaotong [at] mail.neu.edu.cn or heshengmo [at] foxmail.com.

