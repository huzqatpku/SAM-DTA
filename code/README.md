# 说明

该部分代码的主说明文件详见我之前给的代码，本README补充解释更新的内容。

## 代码使用

使用launch_DPI_gnn.py进行多任务（401个蛋白质，每个蛋白质都是一个任务）超参搜索，该代码最终会为每一组超参数调用DPI_GNN_multi_task_main.py进行模型训练。

gnn_configs/GIN_MultiTask_401_4_256_True_0_10_MSELoss.yaml 是launch_DPI_gnn.py搜索得到的最优配置。“401”在这里的意思是401个数据量大于200的蛋白质，在这里每个蛋白质对应多条和药物相关的记录，会进行7：1：2对`记录`划分出train val test。

gnn_configs/GIN_Protein_Graph_280_4_256_True_0_10_MSELoss.yaml 是使用和GIN_MultiTask_401_4_256_True_0_10_MSELoss.yaml一样的超参数配置，但替换训练数据的配置文件，该配置文件使用DPI_GNN_multi_task_main.py进行模型训练，“280”在这里的意思是在`蛋白质级别`划分出的训练集大小，该配置文件会把这280个蛋白质`完整的记录`都拿来训练。

gnn_configs/GIN_Protein_Graph_280_4_256_True_0_10_MSELoss_test.yaml则是用GIN_Protein_Graph_280_4_256_True_0_10_MSELoss.yaml训练好的模型做inference，该配置文件使用DPI_GNN_inference.py运行，运行结果是feats.npy, smiles.txt, lin2.pth，这三个文件我都在群里的百度网盘和压缩文件里提供，它们会移动到 protein_graph/src/data_info目录下。

其他代码如GNN_main.py, GNN_multi_task_launch.py, GNN_multi_task_main.py仅供参考。


## 代码修改 

drug部分在