训练数据：

TrainImage[1, nImage] //训练用图像

TrainShape[nLandmark * 2, nImage] //特征点在图像里的相对位置

RegShape[nLandmark * 2, nImage * nAug(=nSample)] //论文中的S

GtOffset[nLandmark * 2, nSample] //TrainShape和RegShape的偏差

RandomForests[nLandmark] //随机森林

nonzero_leaves[nLandmark, nTree] //当前sample在每棵树中的offset

主要训练流程：

分为若干stage，对于每个stage

1) 计算GtOffset

2) 构造随机森林，森林的数量是nLandmark(fbuild)

2.1) 每个森林通过对应landmark的RegShape和GtOffset构造

2.2) 记录所有森林中叶节点的数量

3) 构造本地特征值，对于每个sample，在所有树内查询，找到在每棵树里的节点编号，得到矩阵Phi[nSample, nLandmark, nTree](frun)

4) 根据论文描述，要计算当前stage的dS，因此对RegShape的每一行计算增量，加到RegShape上，同时导出训练model(ridgeRegression)

主要测试流程：

1) 加载训练得到的随机森林和model

2) 在图上随机采样作为初始landmark

3) 用这些采样点在森林里找到对应叶节点，预测一个值，因为用GtOffset训练，得到的是GtOffset的预测值(liblinear::predict)

4) 把这个GtOffset加到采样点上

5) 进入下一个stage，回到3)

ForestBuild:

采样半径Rmax，按根号2的倍率递减，共10个半径，从大到小循环

1) 圆内随机采400个点，并得到每个点在每个样本里的颜色值

2) 对样本随机索引，用随机索引的样本建立树(tbuild)

3) 用没有被索引到的样本计算方差，遍历所有树，得到offset的总和，和GtOffset计算方差，当方差最小时，该半径作为森林的半径(trun)

ForestRun:

对每棵树作遍历，拿到叶节点的Id和offset(trun)

TreeBuild：

利用传入的索引对应的样本，构建树

1) 如果是个叶节点(深度最大或该节点下只有一个样本)，该叶节点的offset是节点下样本的offset的平均值

2) 利用颜色差作为特征值，数据范围是[-255, 255]

3) 随机挑选m对采样点构建树

3.1) 每对采样点的颜色差作为样本的特征值

3.2) 统计节点下各特征值的样本数，Offset和，Offset平方和，以及方差

3.3) 找到方差下降最大的特征值作为分裂值，记录分裂值，采样点对，以及方差下降值

4) m对采样点处理完后，选择3)中方差下降最大的一组作为最终分裂结果，或是3)中没有得到结果，不分裂，作为叶节点

5) 按照分裂值对样本分为左右两类，分别构建左右子树，当前结点记录采样点对以及分裂值

TreeRun:

利用传入的landmark得到，其对应的叶节点

1) 从根节点开始，利用节点上的采样点信息，在图像中查找颜色，计算特征值

2) 特征值和节点内分裂值比较，小则左子树，大则右子树

3) 循环直到叶节点，返回叶节点的Id和offset

Regression:

1) problem{l = nSample, n = nLeaves, y[nSample] = GtOffsets, x[nSample] = feaNodes, bias = -1}

2) parameter{solver_type = L2R_L2LOSS_SVR, C = 0.5 / lambda, nr_weight = 0, p = 0, eps = 0.001}

3) libliear::train

4) 计算dS(libliear::predict)

problem	l	nSample	
	n	nLeaves	
	y[nSample]	gtOffset	
	x[nSample*landmarkNum*TreeNum]	{leafId+1, 1}	
	bias	-1	
			
parameter	solver	L2R_L2LOSS_SVR	
	C	0.5 / lambda	
	nr_weight	0	
	p	0	
	eps	0.001	
			
train(problem, parameter)	model	nr_feature	nLeaves
		parameter	parameter
		bias	-1
		w[nLeaves]	???
		nr_class	2
		label	NULL
			
train_one(problem, parameter, w, cp=0, cn=0)	fun_obj	fun(double *w)	
		grad(double *w, double *g)	
		Hv(double *s, double *Hs)	
		Xv(double *v, double *Xv)	
		subXv(double *v, double *Xv)	
		subXTv(double *v, double *XTv)	
		double[nSample] C	{0.5 / lambda}
		double[nSample] z	???
		double[nSample] D	???
		int[nSample] I	???
		int sizeI	???
		problem	problem
		double p	0
			
	tron_obj	tron(double *w)	
		trcg(double delta, double *g, double *s, double *r)	
		double norm_inf(int n, double *x)	
		double eps	0.001
		int max_iter	1000
		func_obj	func_obj
			
tron(w)	eta0	0.0001	
	eta1	0.25	
	eta2	0.75	
	sigma1	0.25	
	sigma2	0.5	
	sigma3	4	
	n	nLeaves	
	i	???	
	cg_iter	???	
	delta	???	
	snorm	???	
	alpha	???	
	f	???	
	fnew	???	
	prered	???	
	actred	???	
	gs	???	
	search	1	
	iter	1	
	inc	1	
	s[nLeaves]		
	r[nLeaves]		
	w_new[nLeaves]		
	g[nLeaves]	//梯度	
