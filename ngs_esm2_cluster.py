#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

import esm
import torch
from Bio import SeqIO
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import umap
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------
# 1️⃣ 参数设置
# -----------------------------
fasta_file = "/data02/home/xiaol/ngs_analysis_wtc/NGS/results/clones_full2_aa.fasta"             # NGS AA FASTA 文件
batch_size = 16                         # MODIFIED: 批量大小改大，利用 GPU 加速
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")  # MODIFIED: 统一 device
reduction_method = "umap"               # "pca" 或 "umap"
embedding_dir = "embedding_batches"     # 保存每批 embedding
embedding_file = "all_embeddings.pt"    # MODIFIED: 保存完整 embedding 避免重复计算
os.makedirs(embedding_dir, exist_ok=True)

# -----------------------------
# 2️⃣ 读取序列
# -----------------------------
sequences = []
labels = []
for i, record in enumerate(SeqIO.parse(fasta_file, "fasta")):
    seq = str(record.seq).rstrip("_")  # MODIFIED: 去掉末尾 '_'，防止报错
    sequences.append((record.id, seq))
    labels.append(record.id)
print(f"读取 {len(sequences)} 条序列")

# -----------------------------
# 3️⃣ 加载 t33 模型
# -----------------------------
print("加载 ESM-2 t33_650M 模型 ...")
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.to(device)               # MODIFIED: 移动模型到 GPU
model.eval()
batch_converter = alphabet.get_batch_converter()

# -----------------------------
# 4️⃣ 分批生成 embedding
# -----------------------------
# MODIFIED: 先检查是否已有保存的 embedding，避免重复计算
if os.path.exists(embedding_file):
    print(f"检测到已保存 embedding，直接加载 {embedding_file}")
    saved = torch.load(embedding_file)
    embeddings = saved["embeddings"].numpy()
else:
    all_embeddings = []
    print("开始计算 embeddings ...")
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch)
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[model.num_layers], return_contacts=False)
        token_embeddings = results["representations"][model.num_layers]  # [B, L, D]
        sequence_embeddings = token_embeddings.mean(1)  # 平均残基得到 [B, D]
        sequence_embeddings_cpu = sequence_embeddings.cpu()
        # 保存每批 embedding
        np.save(os.path.join(embedding_dir, f"batch_{i//batch_size}.npy"), sequence_embeddings_cpu.numpy())
        all_embeddings.append(sequence_embeddings_cpu)
        print(f"处理 batch {i//batch_size + 1}/{(len(sequences)-1)//batch_size + 1}")
    # 合并所有 batch
    embeddings = torch.cat(all_embeddings, dim=0)
    # 保存完整 embedding
    torch.save({"labels": labels, "embeddings": embeddings}, embedding_file)
    embeddings = embeddings.numpy()  # 转 numpy 用于降维
print(f"生成 embedding 完成，形状: {embeddings.shape}")

# -----------------------------
# 5️⃣ 降维
# -----------------------------
if reduction_method == "pca":
    reducer = PCA(n_components=50)
    embeddings_reduced = reducer.fit_transform(embeddings)
elif reduction_method == "umap":
    pca = PCA(n_components=50)
    embeddings_pca = pca.fit_transform(embeddings)
    reducer = umap.UMAP(n_components=2, random_state=42)
    embeddings_reduced = reducer.fit_transform(embeddings_pca)
else:
    raise ValueError("reduction_method 必须是 'pca' 或 'umap'")
print("降维完成")

# -----------------------------
# 6️⃣ HDBSCAN 聚类
# -----------------------------
clusterer = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=5)
labels_cluster = clusterer.fit_predict(embeddings_reduced)
print(f"聚类完成，共 {len(np.unique(labels_cluster))} 个类（-1 为噪声）")

# -----------------------------
# 7️⃣ 可视化
# -----------------------------
plt.figure(figsize=(8,6))
palette = sns.color_palette("tab20", n_colors=len(np.unique(labels_cluster)))
sns.scatterplot(
    x=embeddings_reduced[:,0],
    y=embeddings_reduced[:,1],
    hue=labels_cluster,
    palette=palette,
    legend="full",
    s=10
)
plt.title("NGS 序列 ESM-2 t33 Embedding + 降维 + HDBSCAN 聚类")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# -----------------------------
# 8️⃣ 保存聚类结果
# -----------------------------
df_out = pd.DataFrame({
    "seq_id": [s[0] for s in sequences],
    "sequence": [s[1] for s in sequences],
    "cluster": labels_cluster
})
df_out.to_csv("ngs_sequences_clusters.csv", index=False)
print("聚类结果保存为 ngs_sequences_clusters.csv")
