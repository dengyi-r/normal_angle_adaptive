#include"Registration.h"

Eigen::Matrix4f Registration::RegistrationTransform(
	PointCloud::Ptr& key_src, 
//	pcl::PointCloud<pcl::ShapeContext1980>::Ptr sps_src,
	PointCloud::Ptr& key_tgt, 
//	pcl::PointCloud<pcl::ShapeContext1980>::Ptr sps_tgt
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs_tgt,
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs_src
)
{
	clock_t start = clock();
	//RANSAC配准
	pcl::console::print_highlight("开始进行配准\n");
	pcl::console::print_highlight("开始进行的源配准\n");
	std::cout << key_src->size() << endl;
	pcl::console::print_highlight("开始进行目标配准\n");
	std::cout << key_tgt->size() << endl;
	pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> ransac;
	ransac.setInputSource(key_src);
//	ransac.setSourceFeatures(sps_src);
	ransac.setSourceFeatures(fpfhs_src);
	ransac.setInputTarget(key_tgt);
//	ransac.setTargetFeatures(sps_tgt);
	ransac.setTargetFeatures(fpfhs_tgt);
	ransac.setMaximumIterations(m_RansacIterations); //  采样一致性迭代次数
	ransac.setNumberOfSamples(200);                    //  创建假设所需的样本数
//	ransac.setCorrespondenceRandomness(10);          //  使用的临近特征点的数目

	ransac.setMaxCorrespondenceDistance(m_RansacDistance); // 判断是否为内点的距离阈值
//	ransac.setInlierFraction(m_Fraction);                  //  接受位姿假设所需的内点比例

	PointCloud::Ptr sac_result(new PointCloud);
	ransac.align(*sac_result);

	Eigen::Matrix4f ransac_trans;
	ransac_trans = ransac.getFinalTransformation();
	clock_t sac_time = clock();

	//icp配准
	PointCloud::Ptr icp_result(new PointCloud);
	pcl::IterativeClosestPoint<PointT, PointT> icp;
	icp.setInputSource(key_src);
	icp.setInputTarget(key_tgt);

//	icp.setMaxCorrespondenceDistance(m_MaxDistance);
	icp.setMaximumIterations(m_MaxIterations);        // 最大迭代次数
	icp.setTransformationEpsilon(1e-10); // 两次变化矩阵之间的差值
	icp.setEuclideanFitnessEpsilon(0.001);// 均方误差


	icp.align(*icp_result, ransac_trans);
	clock_t end = clock();
	std::cout << "total time: " << (double)(end - start) / (double)CLOCKS_PER_SEC << " s" << std::endl;
	std::cout << "sac time: " << (double)(sac_time - start) / (double)CLOCKS_PER_SEC << " s" << std::endl;
	std::cout << "icp time: " << (double)(end - sac_time) / (double)CLOCKS_PER_SEC << " s" << std::endl;

	std::cout << "ICP has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;
	Eigen::Matrix4f icp_trans;

	Eigen::Matrix4f FinalTrans = icp.getFinalTransformation();

	return FinalTrans;
}

void  Registration::VisualizeRegistration(PointCloud::Ptr& source, PointCloud::Ptr& target, PointCloud::Ptr& icp, PointCloud::Ptr target_key, PointCloud::Ptr source_key)
{
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("RegistrationCloud"));
	int v1 = 0;
	int v2 = 1;
	viewer->createViewPort(0, 0, 0.5, 1, v1);
	viewer->createViewPort(0.5, 0, 1, 1, v2);
	viewer->setBackgroundColor(0, 0, 0, v1);
	viewer->setBackgroundColor(0.05, 0, 0, v2);

	//原始点云绿色
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> src_h(source, 0, 255, 0);
	//目标点云蓝色
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> tgt_h(target, 0, 0, 255);
	//转换后的源点云红色
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> transe(icp, 255, 0, 0);
	//viewer->setBackgroundColor(255, 255, 255);
	viewer->addPointCloud(source, src_h, "source cloud", v1);
	viewer->addPointCloud(target, tgt_h, "target cloud", v1);

	viewer->addPointCloud(target, tgt_h, "target cloud1", v2);
	viewer->addPointCloud(icp, transe, "pcs cloud", v2);

	//---------------------------关键点提取可视化-------------------------------
	int v3 = 0;
	int v4 = 1;
	
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_iss(new pcl::visualization::PCLVisualizer("Key_point"));
	viewer_iss->createViewPort(0, 0, 0.5, 1, v3);
	viewer_iss->createViewPort(0.5, 0, 1, 1, v4);
	viewer_iss->setBackgroundColor(0, 0, 0, v3);
	viewer_iss->setBackgroundColor(0.05, 0, 0, v4);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color_point(target_key, 255,255,0);
	viewer_iss->addPointCloud(target_key, target_color_point, "key_target cloud", v3 );
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_color_point(source_key, 255, 0, 255);
	viewer_iss->addPointCloud(source_key, source_color_point, "key_source cloud", v4 );

	//添加坐标系
//	viewer->addCoordinateSystem(10);
//	viewer_iss->addCoordinateSystem(10);
	//viewer->initCameraParameters();
	while (!viewer->wasStopped())
	{
		viewer_iss->spinOnce(100);
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(10000));
	}

}

void Registration::Nicp_reg(
	PointCloud::Ptr& key_src,
	PointCloud::Ptr& key_tgt,
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs_tgt,
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs_src
) {

}

void Registration::geometry_sampling(
	PointCloud::Ptr& incloud,
	PointCloud::Ptr& outcloud) {


	//-----------------计算点云的法线--------------------------
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> nor_omp;
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	//建立kdtree进行近邻点搜索
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	nor_omp.setNumberOfThreads(8);
	nor_omp.setInputCloud(incloud);
	nor_omp.setSearchMethod(tree);
	nor_omp.setKSearch(20);
	nor_omp.compute(*normals);

	/*
	//----------------点云的主曲率计算--------------------------
	pcl::PrincipalCurvaturesEstimation<pcl::PointXYZ, pcl::Normal, pcl::PrincipalCurvatures> Pri;
	Pri.setInputCloud(incloud);
	Pri.setInputNormals(normals);
	Pri.setSearchMethod(tree);
	Pri.setKSearch(20);
	//计算点云主曲率
	pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr pri_normal;
	Pri.compute(*pri_normal);

	//显示和检索第0点的主曲率
	cout << "最大主曲率：" << pri_normal->points[0].pc1 << endl;//输出最大主曲率
	cout << "最大主曲率：" << pri_normal->points[0].pc2 << endl;//输出最大主曲率
	//主曲率方向（最大特征值对应的特征向量）
	cout << "主曲率方向：" << endl;
	cout << pri_normal->points[0].principal_curvature_x << endl;
	cout << pri_normal->points[0].principal_curvature_y << endl;
	cout << pri_normal->points[0].principal_curvature_z << endl;
	*/

	//--------------根据法线夹角计算------------------

	float angle_1 = 0.0;
	float average_sum_anglek = 0.0; //定义邻域内k个法向量夹角的平均值
	std::vector<int> indexes;

	float threshold = 5;// 夹角阈值

	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(incloud);//设置需要建立kdtree的点云指针
	int k = 25;//设置需要查找的临近点个数
	std::vector<int> pointIndxNKNSearch(k);//保存每个近邻点索引
	std::vector<float> pointNKNSpuareDistance(k);//保存每个近邻点与查找点之间的欧式距离平方
	pcl::PointXYZ searchPoint;
	for (size_t i = 0; i < incloud->points.size(); i++)
	{
		if (kdtree.nearestKSearch(searchPoint, k, pointIndxNKNSearch, pointNKNSpuareDistance) > 0) {

			float sum_angle = 0.0; //定义k个邻近的点法向量夹角之和

			for (size_t m = 0; m < pointIndxNKNSearch.size(); ++m)
			{
				/*计算法向量的夹角*/
				Eigen::Vector3f v1(normals->points[i].data_n[0],
					normals->points[i].data_n[1],
					normals->points[i].data_n[2]
					),
					v2(normals->points[m].data_n[0],
						normals->points[m].data_n[1],
						normals->points[m].data_n[2]
					);
				//计算夹角（法一：调用库函数）
				angle_1 = pcl::getAngle3D(v1, v2, true);  //bool in_degree = false
				//法二
				//double radian_angle = atan2(v1.cross(v2).norm, v1.transpose() * v2); //[0,pi]
				//angle_1 = radian_angle;

			}
			sum_angle += angle_1;
			average_sum_anglek = sum_angle / pointIndxNKNSearch.size();//邻域夹角均值
			//----------------提取特征点---------------
			if (average_sum_anglek > threshold)
			{
				indexes.push_back(i);
			}
		}
	}
	pcl::copyPointCloud(*incloud, indexes, *outcloud);
	cout << "几何采样后的点云数: " << outcloud->size() << endl;





}
