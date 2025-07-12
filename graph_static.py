import networkx as nx
import numpy as np
from collections import defaultdict
import pandas as pd


def gini_coefficient(degrees):
    """Calculate the Gini coefficient of a list of degrees."""
    # 将度数列表转换为numpy数组
    degrees = np.array(degrees,dtype=np.int64)
    
    # Mean absolute difference
    mad = np.abs(np.subtract.outer(degrees, degrees)).mean()
    # Relative mean absolute difference
    rmad = mad / np.mean(degrees)
    # Gini coefficient
    g = 0.5 * rmad
    return g


#关系类型不平衡指数
def Relationship_types_distribution_index(relation_counts):
    """Calculate the imbalance index of relation frequencies using Gini coefficient."""
    # 获取每种关系类型的频率（值）
    frequencies = list(relation_counts.values())
    # 使用 Gini 系数来衡量频率分布的平衡度
    return gini_coefficient(frequencies)

#关系类型不平衡指数
def count_relation_cats_distribution(G):
    """Count different relation types: 1-1, 1-n, n-1, n-n."""
    s11 = s1n = sn1 = snn = 0

    # 遍历图中的所有边（关系三元组）
    for u, v, data in G.edges(data=True):
        relation = data.get('relation')
        if relation is None:
            continue

        # 获取关系的左右两个节点
        left_neighbors = set(G.predecessors(u))  # u的入邻居
        right_neighbors = set(G.successors(v))  # v的出邻居

        rign = len(left_neighbors) / len(set(G.neighbors(u))) if len(set(G.neighbors(u))) > 0 else 0
        lefn = len(right_neighbors) / len(set(G.neighbors(v))) if len(set(G.neighbors(v))) > 0 else 0

        # 判断该关系属于哪种类型并更新计数
        if rign < 1.5 and lefn < 1.5:
            s11 += 1
        elif rign >= 1.5 and lefn < 1.5:
            s1n += 1
        elif rign < 1.5 and lefn >= 1.5:
            sn1 += 1
        elif rign >= 1.5 and lefn >= 1.5:
            snn += 1

    return s11, s1n, sn1, snn
def relation_cat_imbalance_index(s11, s1n, sn1, snn):
    """Calculate the imbalance index for relation type distribution."""
    counts = [s11, s1n, sn1, snn]
    return gini_coefficient(counts)





def count_relation_types(G):
    """Count the number of edges for each relation type in a MultiDiGraph."""
    # 用于存储每种关系类型的边的数量
    relation_counts = defaultdict(int)
    
    # 遍历图中的所有边，获取每条边的属性"relation"
    for u, v, data in G.edges(data=True):
        relation = data.get('relation')
        if relation is not None:
            relation_counts[relation] += 1
    
    return relation_counts

def compute_multidigraph_density(G):
    """Compute the density of a MultiDiGraph."""
    if not isinstance(G, nx.MultiDiGraph):
        raise ValueError("The input graph must be a MultiDiGraph.")
    
    n = len(G.nodes())  # 节点数
    if n <= 1:
        return 0  # 如果只有一个节点，密度为0
    
    # 实际存在的边数（包括多重边）
    actual_edges = G.number_of_edges()
    
    # 有向图中最多的边数（不考虑多重边）
    possible_edges = n * (n - 1)  # 计算不包括自环的可能最大边数
    
    # 计算密度
    density = actual_edges / possible_edges
    
    return density

def global_clustering_coefficient_for_multidigraph(G):
    """
    Calculate the global clustering coefficient of a MultiDiGraph by converting it into an undirected graph.
    """
    # 将 MultiDiGraph 转换为无向图
    undirected_G = nx.Graph(G)  

    # 使用 networkx 计算无向图的全局聚类系数
    clustering_coefficient = nx.transitivity(undirected_G)  # transitivity = global clustering coefficient

    return clustering_coefficient

def directed_global_clustering_coefficient(G):
    """
    Calculate the global clustering coefficient for a directed MultiDiGraph.
    """
    # 统计所有可能的三元组（两条边连接三个节点，不一定闭合）
    triples = 0
    # 统计闭合三元组（三角形）
    triangles = 0

    for node in G.nodes():
        # 获取当前节点的入邻居和出邻居
        neighbors_in = set(G.predecessors(node))
        neighbors_out = set(G.successors(node))
        
        # 计算当前节点的所有入邻居和出邻居的组合
        for v in neighbors_out:
            for u in neighbors_in:
                if G.has_edge(v, u):  # 检查是否有 v -> u 的边，即形成三角形
                    triangles += 1
                triples += 1  # 每次形成一个三元组
    
    if triples == 0:
        return 0
    
    # 全局聚类系数 = 闭合三元组数 / 总三元组数
    clustering_coefficient = triangles / triples
    return clustering_coefficient

def graph_charater(G,sample_ratio):

    degree_centrality = nx.degree_centrality(G)
    # 计算平均度中心性
    average_degree_centrality = sum(degree_centrality.values()) / len(degree_centrality)
    
    # 使用基尼系数计算Degree Distribution Index
    degrees = [d for n, d in G.degree()]
    degree_distribution_index = gini_coefficient(degrees)

    # 计算关系类型（1-1,1-n）等不均衡指数
    s11, s1n, sn1, snn = count_relation_cats_distribution(G)
    Relationship_categories_distribution_index = relation_cat_imbalance_index(s11, s1n, sn1, snn)

    #计算关系类型不均衡指数
    rel_type_count = count_relation_types(G)
    rel_c = [c for c in rel_type_count.values()]
    print(rel_c)


    rel_type_imbalance_index = Relationship_types_distribution_index(rel_type_count)
    print(rel_type_imbalance_index)




    #计算图密度
    density = compute_multidigraph_density(G)

    #计算全局聚类系数
    global_clustering = directed_global_clustering_coefficient(G)

    DG = nx.DiGraph(G)  # G 为 MultiDiGraph

    # 计算强连通分量
    strongly_connected_components = len(list(nx.strongly_connected_components(DG)))
    # 计算弱连通分量
    weakly_connected_components = len(list(nx.weakly_connected_components(DG)))

    # # 将 MultiDiGraph 转为无向图，并计算近似直径
    # UG = G.to_undirected()
    # approx_diameter = nx.diameter(UG.subgraph(max(nx.connected_components(UG), key=len)))

    # 计算实体数、关系数、关系类型数
    num_entities = G.number_of_nodes()
    num_relations = G.number_of_edges()
    num_relation_types = len(set(nx.get_edge_attributes(G, 'relation').values()))







    print("degree_distribution_index:",degree_distribution_index)

    print("Relationship_categories_distribution_index:", Relationship_categories_distribution_index)
    print("rel_type_imbalance_index:", rel_type_imbalance_index)

    print("Graph density:",density)
    print("Global Clustering Coefficient:", global_clustering)
    print("strongly_connected_components:", strongly_connected_components)



    # return {
    #     "average_degree_centrality": average_degree_centrality,
    #     "degree_distribution_index": degree_distribution_index,
    #     "relType_imbalance_index": relType_imbalance_index,
    #     "Graph_density": density,
    #     "Global_clustering_coefficient": global_clustering,
    #     "strongly_connected_components": strongly_connected_components,
    #     "weakly_connected_components": weakly_connected_components,
    #     "approx_diameter": approx_diameter,
    #     "Number of entities": num_entities,
    #     "Number of relations:": num_relations,
    #     "Number of relation types": num_relation_types
    # }
    return {
        "degree_distribution_index": degree_distribution_index,
        "Relationship_categories_distribution_index": Relationship_categories_distribution_index,
        "rel_type_imbalance_index": rel_type_imbalance_index, 
        "Graph_density": density,
        "Global_clustering_coefficient": global_clustering,
        "strongly_connected_components": strongly_connected_components,

    }
import random
if __name__ == "__main__":

    G = nx.MultiDiGraph()
    with open('.\train.txt', 'r') as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            G.add_edge(h, t, relation=r)
    results_df = pd.DataFrame()
    n = 1
    for i in range(n):
        # 运行采样后的图统计计算
        sample_ratio = random.uniform(1.0,1.0)
        stats = graph_charater(G,sample_ratio)
        stats_df = pd.DataFrame([stats])

        # 将结果转换为DataFrame的一行
        results_df = pd.concat([results_df,stats_df], ignore_index=True)

    # 将结果保存到Excel文件中
    results_df.to_excel("origin.xlsx", index=False)
    












