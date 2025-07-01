from BCEmbedding import RerankerModel


docs = [
        # 技术领域
        "OpenVINO™ is Intel’s comprehensive open-source toolkit for optimizing and deploying AI inference across CPUs, GPUs, and other accelerators. It supports ONNX, PyTorch, and TensorFlow models.",
        "PyTorch is a Python-based deep learning framework with dynamic computation graphs, popular for research and natural language processing.",
        "TensorFlow is Google’s end-to-end ML platform supporting both training and inference, with strong production deployment capabilities.",
        "Hugging Face Transformers simplifies using state-of-the-art NLP models and provides tools for fine-tuning and deployment.",
        "Intel’s Arc A770 is a discrete GPU designed for gaming and content creation, leveraging Xe-HPG architecture and supporting OpenVINO acceleration.",
        
        # 文学领域
        "To Kill a Mockingbird (1960) by Harper Lee explores racial injustice through the eyes of young Scout Finch. It won the Pulitzer Prize.",
        "Herman Melville’s Moby-Dick (1851) is a philosophical novel about Captain Ahab’s obsession with hunting a white whale.",
        "Jane Austen’s Pride and Prejudice (1813) examines love and class through the Bennet sisters’ relationships.",
        "J.K. Rowling’s Harry Potter series follows a young wizard’s journey through seven books, becoming a global cultural phenomenon.",
        
        # 混淆项和歧义
        "Python (programming language) was created by Guido van Rossum in 1991, emphasizing readability and simplicity.",
        "Python (snake) is a genus of nonvenomous reptiles found in tropical regions, known for constricting prey.",
        "Monty Python was a British comedy group active in the 1970s, famous for their surreal sketches and films like 'Monty Python and the Holy Grail'.",
        "ARC (Animation Research Council) is a fictional organization in the video game 'Horizon Zero Dawn', unrelated to Intel’s GPU.",
        "OpenAI’s GPT-4 is a large language model capable of generating human-like text, while OpenVINO focuses on inference optimization.",
        
        # 跨领域关联
        "Intel’s OpenVINO can accelerate PyTorch and TensorFlow models by up to 10x on Arc GPUs, enabling real-time applications like video analytics.",
        "Fine-tuning Hugging Face models with PyTorch and deploying them with OpenVINO on edge devices reduces latency and resource usage.",
    ]
queries = [
        # 直接事实查询
        "Who is the author of To Kill a Mockingbird?",
        "What is OpenVINO’s primary purpose?",
        "Which company developed TensorFlow?",
        
        # 多义词歧义
        "What is Python?",  # 编程 vs. 蛇
        "Who are Monty Python?",  # 喜剧组 vs. 编程语言
        
        # 细节对比
        "How does OpenVINO differ from TensorFlow?",
        "What are the key features of PyTorch compared to Hugging Face Transformers?",
        
        # 跨文档推理
        "Can OpenVINO optimize Hugging Face models?",
        "What hardware can run OpenVINO-accelerated models?",
        
        # 混淆项测试
        "Is ARC related to Intel’s GPU technology?",
        "What’s the connection between OpenVINO and Arc GPUs?",
        
        # 时效性和错误信息
        "When was To Kill a Mockingbird published?",  # 测试是否返回1960而非干扰年份
        "Does OpenVINO support TensorFlow models?",  # 验证是否优先选择明确支持的文档
    ]

# your query and corresponding passages
query = queries[0]
passages = docs

# construct sentence pairs
sentence_pairs = [[query, passage] for passage in passages]

# init reranker model
model = RerankerModel(use_ov = True, model_name_or_path="maidalun1020/bce-reranker-base_v1", ov_model_path = "./bce-reranker-base_v1-ov", ov_device = "GPU")

# method 0: calculate scores of sentence pairs
scores = model.ov_compute_score(sentence_pairs)

# method 1: rerank passages
rerank_results = model.rerank(query, passages)


for doc, score in zip(rerank_results['rerank_passages'], rerank_results['rerank_scores']): print(f"{score:.4f}\t{doc}")
