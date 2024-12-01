# GNN-Tutorial
# Graph Neural Network Tutorial for Deep Learning (CO460)

## 1. Pytorch Geometric Framework

In this section, we introduce the **Pytorch Geometric** (PyG) framework, which is designed for deep learning on graph-structured data. PyG simplifies the implementation of Graph Neural Networks (GNNs) and allows us to focus on developing models while handling the intricacies of graph data representation and processing.

### Key Topics:

- **Message Passing Scheme**: The foundation of most GNNs is the message-passing mechanism, where information is propagated through the graph. PyG provides a flexible framework for defining and running message-passing operations.
- **Efficient Graph Representations**: PyG offers optimized data structures such as `Data` and `Batch` for representing graph data and minibatching graphs for efficient training.
- **Graph Convolution Networks (GCN)**: This tutorial will walk you through the implementation of **Graph Convolution Networks (GCN)** based on the seminal paper by Kipf and Welling, ["SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS"](https://arxiv.org/abs/1609.02907). We will also demonstrate how to implement **GraphSAGE** based on the message-passing scheme.
  
  - **GraphSAGE** (Hamilton et al., 2017) introduces inductive representation learning on large graphs, which is key for handling graphs with new, unseen nodes during training. You will implement different aggregation functions (mean, sum, max) in GraphSAGE to understand their effects on model performance.

## 2. Vertex Classification

In this section, you will use the concepts learned from the **Pytorch Geometric** framework to build a vertex classification model. Vertex classification involves predicting the labels of nodes in a graph based on node features and graph structure.

### Tasks:

- **GCN for Vertex Classification**: We will implement a GCN model and apply it to the **Cora dataset**, which is a citation network where each node represents a scientific paper, and edges represent citation relationships. The goal is to classify nodes (papers) into predefined categories based on their features and graph structure.
  
- **GraphSAGE for Vertex Classification**: In addition to GCN, we will implement **GraphSAGE** (Hamilton et al., 2017) to perform vertex classification using different aggregation functions (mean, sum, max). By comparing the performance of these methods, you will gain insights into how aggregation functions impact the quality of the model’s predictions.

## 3. Graph Classification

This section introduces **Graph Classification**, where the task is to classify entire graphs rather than individual vertices.

### Tasks:

- **GINConv for Graph Classification**: We will implement **Graph Isomorphism Network (GIN)** Conv layers, based on the paper ["HOW POWERFUL ARE GRAPH NEURAL NETWORKS?"](https://arxiv.org/abs/1810.00826) by Xu et al. GINConv is known for its power in distinguishing graphs and is particularly effective for graph classification tasks.
  
- **Benchmarking on IMDB Dataset**: We will use the **IMDB dataset** for graph classification, where each graph represents a movie and the goal is to classify movies into genres. You will experiment with different aggregation functions (SUM, MEAN, MAX) to evaluate their effect on the performance of the graph classification model.

## 4. Running the Code

To run the code, follow these steps:

1. **Install Dependencies**:
   Ensure that you have installed the necessary dependencies, including **PyTorch** and **PyTorch Geometric**. You can install them using the following commands:
   ```bash
   pip install torch
   pip install torch-geometric
