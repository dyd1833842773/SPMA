import os
import random
import networkx as nx
import numpy as np
from graph_static import graph_charater
import pandas as pd


def load_triples_from_file(file_path):
    """Load triples from a file and return a list of (head, relation, tail) tuples."""
    triples = []
    with open(file_path, 'r') as file:
        for line in file:
            head, relation, tail = line.strip().split('\t')
            triples.append((head, relation, tail))
    return triples

def build_graph_from_triples(triples):
    """Build a directed graph from triples."""
    G = nx.MultiDiGraph()
    for head, relation, tail in triples:
        G.add_edge(head, tail, relation=relation)
    return G




# def sample_connected_subgraph(G, sample_ratio, top_k=50):
#     """Sample a connected subgraph from graph G with a given sample ratio."""
#     target_node_count = int(len(G) * sample_ratio)
    
#     # 对节点按度数排序，获取度数最高的 top_k 个节点
#     top_degree_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:top_k]
#     top_degree_nodes = [node for node, degree in top_degree_nodes]
    
#     # 从这些节点中随机选择一个初始节点
#     start_node = random.choice(top_degree_nodes)
    
#     # 使用BFS确保连通
#     sampled_nodes = set()
#     queue = [start_node]
#     sampled_nodes.add(start_node)
    
#     while queue and len(sampled_nodes) < target_node_count:
#         current_node = queue.pop(0)
        
#         # 遍历该节点的邻居节点
#         for neighbor in G.neighbors(current_node):
#             if neighbor not in sampled_nodes:
#                 queue.append(neighbor)
#                 sampled_nodes.add(neighbor)

#     # 创建子图
#     subgraph = G.subgraph(sampled_nodes).copy()

#     return subgraph
def sample_connected_subgraph(G, max_attempts=100):
    """Sample a connected subgraph from graph G with a given sample ratio. If sampling fails, retry up to max_attempts times."""
    
    attempts = 0
    
    while attempts < max_attempts:
        # 每次采样的随机比例
        sample_ratio = random.uniform(0.5, 1.0)

        print("sample_ratio",sample_ratio)
        target_node_count = int(len(G) * sample_ratio)
        # 对节点按度数排序，选择度数最高的 top_k 个节点中随机选择一个起始点
        top_k = 50  # 可以调整 top_k 值以覆盖高连接度节点
        sorted_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:top_k]
        start_node = random.choice([node for node, degree in sorted_nodes])
        
        # 使用 BFS 确保连通
        sampled_nodes = set()
        queue = [start_node]
        sampled_nodes.add(start_node)
        
        while queue and len(sampled_nodes) < target_node_count:
            current_node = queue.pop(0)
            
            # 遍历该节点的邻居节点
            for neighbor in G.neighbors(current_node):
                if neighbor not in sampled_nodes:
                    queue.append(neighbor)
                    sampled_nodes.add(neighbor)
        
        # 判断采样是否达到预期规模
        if len(sampled_nodes) >= target_node_count:
            subgraph = G.subgraph(sampled_nodes).copy()
            return subgraph,sample_ratio
        
        attempts += 1
        print(f"Retrying... Attempt {attempts}/{max_attempts}")
    
    # 如果达到最大尝试次数仍未成功，返回 None 或报错
    print("Sampling failed after maximum attempts.")
    return None

def subgraph_to_triples(subgraph):
    """Convert the sampled subgraph into triples."""
    triples = []
    for u, v, data in subgraph.edges(data=True):
        relation = data['relation']
        triples.append((u, relation, v))
    random.shuffle(triples)
    return triples

def create_entity_relation_mappings(triples):
    """Create new entity2id and relation2id mappings."""
    sampled_entities = set()
    sampled_relations = set()
    
    for head, relation, tail in triples:
        sampled_entities.update([head, tail])
        sampled_relations.add(relation)
    
    new_entity2id = {entity: idx for idx, entity in enumerate(sorted(sampled_entities))}
    new_relation2id = {relation: idx for idx, relation in enumerate(sorted(sampled_relations))}
    
    return new_entity2id, new_relation2id

def convert_triples_to_ids(triples, entity2id, relation2id):
    """Convert triples to their corresponding IDs using new_entity2id and new_relation2id."""
    triples_with_ids = []
    for head, relation, tail in triples:
        triples_with_ids.append((entity2id[head], relation2id[relation], entity2id[tail]))
    return triples_with_ids

def filter_triples_by_subgraph(triples, entity2id, relation2id):
    """Filter triples based on the subgraph's entity and relation ID mappings."""
    filtered_triples_id = []
    filtered_triples = []
    for head, relation, tail in triples:
        if head in entity2id and tail in entity2id and relation in relation2id:
            filtered_triples_id.append((entity2id[head], relation2id[relation], entity2id[tail]))
            filtered_triples.append((head, relation, tail))
    return filtered_triples_id,filtered_triples

def filter_triples_by_subgraph_ow(triples, entity2id, relation2id):
    """Filter triples based on the subgraph's entity and relation ID mappings."""
    filtered_triples = []
    for head, relation, tail in triples:
        if tail in entity2id and relation in relation2id:
            filtered_triples.append((head, relation, tail))
    return filtered_triples

def save_triples_id(triples, file_path):
    """Save triples to a file."""
    with open(file_path, 'w') as file:
        file.write(f"{len(triples)}\n")
        for head, relation, tail in triples:
            file.write(f"{head}\t{tail}\t{relation}\n")     #注意train2id.txt的实体关系顺序

def save_triples(triples, file_path):
    """Save triples to a file."""
    with open(file_path, 'w') as file:
        for head, relation, tail in triples:
            file.write(f"{head}\t{relation}\t{tail}\n")


def save_mapping(mapping, file_path, len_in_head = True):
    """Save entity2id or relation2id mapping to a file."""
    with open(file_path, 'w') as file:
        if len_in_head:
            file.write(f"{len(mapping)}\n")
        for key, value in mapping.items():
            file.write(f"{key}\t{value}\n")

def perform_sampling(start,end, seed,train_file, test_file, valid_file,output_dir,output_samples_not_openke):
    """Perform n random samplings and save the results in separate directories."""
    random.seed(seed)
    np.random.seed(seed)
    triples_train = load_triples_from_file(train_file)
    triples_test = load_triples_from_file(test_file)
    triples_valid = load_triples_from_file(valid_file)

    '''triples_test_zero = load_triples_from_file(test_file_zero)
    triples_valid_zero = load_triples_from_file(valid_file_zero)'''


    results_df = pd.DataFrame()

    G = build_graph_from_triples(triples_train)
    
    for i in range(start,end+1):
        # # 每次采样的随机比例
        # sample_ratio = random.uniform(0.5, 1.0)

        # print("sample_ratio",sample_ratio)
        
        # 采样连通子图
        subgraph, sample_ratio = sample_connected_subgraph(G)

        # 运行采样后的图统计计算
        stats = graph_charater(subgraph, sample_ratio)
        stats_df = pd.DataFrame([stats])

        # 将结果转换为DataFrame的一行
        results_df = pd.concat([results_df,stats_df], ignore_index=True)
        # 将结果转换为DataFrame的一行
        #results_df = results_df.append(stats, ignore_index=True)
        
        # 将子图转换为三元组
        sampled_triples_train = subgraph_to_triples(subgraph)
        
        # 创建新的实体和关系ID映射
        new_entity2id, new_relation2id = create_entity_relation_mappings(sampled_triples_train)
        
        # 将三元组转换为ID格式
        triples_with_ids_train = convert_triples_to_ids(sampled_triples_train, new_entity2id, new_relation2id)
        
        # 过滤测试集和验证集
        filtered_triples_test_id,filtered_triples_test = filter_triples_by_subgraph(triples_test, new_entity2id, new_relation2id)
        filtered_triples_valid_id,filtered_triples_valid = filter_triples_by_subgraph(triples_valid, new_entity2id, new_relation2id)
        
        # 保存结果到sample_n目录
        sample_output_dir = os.path.join(output_dir, f"sample_{i}")
        os.makedirs(sample_output_dir, exist_ok=True)
        
        save_triples_id(triples_with_ids_train, os.path.join(sample_output_dir, 'train2id.txt'))
        #save_triples(sampled_triples_train, os.path.join(sample_output_dir, 'train.txt'))
        save_mapping(new_entity2id, os.path.join(sample_output_dir, 'entity2id.txt'))
        save_mapping(new_relation2id, os.path.join(sample_output_dir, 'relation2id.txt'))
        save_triples_id(filtered_triples_test_id, os.path.join(sample_output_dir, 'test2id.txt'))
        save_triples_id(filtered_triples_valid_id, os.path.join(sample_output_dir, 'valid2id.txt'))
        # save_triples(filtered_triples_test, os.path.join(sample_output_dir, 'test.txt'))
        # save_triples(filtered_triples_valid, os.path.join(sample_output_dir, 'valid.txt'))

        '''# 同理保存一份结果用于ow-lp
        # 过滤测试集和验证集
        filtered_triples_test_zero = filter_triples_by_subgraph_ow(triples_test_zero, new_entity2id, new_relation2id)
        filtered_triples_valid_zero = filter_triples_by_subgraph_ow(triples_valid_zero, new_entity2id, new_relation2id)
        sample_output_dir_ow = os.path.join(output_dir_ow, f"sample_{i}")
        os.makedirs(sample_output_dir_ow, exist_ok=True)
        save_triples(sampled_triples_train, os.path.join(sample_output_dir_ow, 'train.txt'))
        save_triples(filtered_triples_test_zero, os.path.join(sample_output_dir_ow, 'test_zero.txt'))
        save_triples(filtered_triples_valid_zero, os.path.join(sample_output_dir_ow, 'valid_zero.txt'))
        save_mapping(new_entity2id, os.path.join(sample_output_dir_ow, 'entity2id.txt'),len_in_head = False)
        save_mapping(new_relation2id, os.path.join(sample_output_dir_ow, 'relation2id.txt'),len_in_head = False)
'''
        # 同理保存train.txt 到另一个文件夹
        sample_output_dir_not_openke = os.path.join(output_samples_not_openke, f"sample_{i}")
        os.makedirs(sample_output_dir_not_openke, exist_ok=True)
        save_triples(sampled_triples_train, os.path.join(sample_output_dir_not_openke, 'train.txt'))
        save_triples(filtered_triples_test, os.path.join(sample_output_dir_not_openke, 'test.txt'))
        save_triples(filtered_triples_valid, os.path.join(sample_output_dir_not_openke, 'valid.txt'))

    # 将结果保存到Excel文件中
    excel_output_path = os.path.join(output_dir, f"sampling_results_{i}.xlsx")
    results_df.to_excel(excel_output_path, index=False)

if __name__ == "__main__":
    train_file = '.\train.txt'  
    test_file = '.\test.txt'    
    valid_file = '.\valid.txt' 



    output_dir = r'18rr\output1'  # 输出目录
    output_samples_not_openke = r'18rr\output2'
    start = 51
    end = 52
    n = end-start+1  # 采样次数
    seed = 43  # 随机数种子
    perform_sampling(start,end,seed,train_file, test_file, valid_file, output_dir,output_samples_not_openke)

    ##1-20 随机数种子42   数据文件sampling_results_20.xlsx
    ##21-50 随机数种子43  数据文件sampling_results_50.xlsx