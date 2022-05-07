#include <iostream>
#include <vector>
#include "Registration.h"
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
//#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Core>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/registration/ia_kfpcs.h> //K4PCS算法头文件
#include <pcl/segmentation/segment_differences.h>
#include <pcl/segmentation/region_growing.h> //区域生长
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/filters/voxel_grid.h>

#include <vtkAutoInit.h>

VTK_MODULE_INIT(vtkInteractionStyle);
VTK_MODULE_INIT(vtkRenderingFreeType);

using namespace std;

//存放每一个点的索引值和其对应的曲率
typedef struct PCURVATURE {
	int index;
	float curvature;
}PCURVATURE;

//存放点云的每一个点、index及对应点的rmse，
typedef struct point_all
{
	int index;
	pcl::PointXYZ point_data;
	float rmse_point;
	float pointTodistance_1;
};

//获取point_all集 寻找target的对应点及rmse
std::vector<point_all> point_data_all_vec;
point_all point_data_all;

//寻找source的对应点及rmse
std::vector<point_all> point_data_all_vec_1;
point_all point_data_all_1;


void downingSampling(pcl::PointCloud<pcl::PointXYZ>::Ptr  pcd_src,
	pcl::PointCloud<pcl::PointXYZ>::Ptr  pcd_tgt,
	double LeafSize) {

	//下采样滤波
	pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
	//	double LeafSize = 2;  //pcd 0.003
	voxel_grid.setLeafSize(LeafSize, LeafSize, LeafSize);
	voxel_grid.setInputCloud(pcd_src);
	voxel_grid.filter(*pcd_tgt);
	std::cout << "down size *cloud_src_o from " << pcd_src->size() << "to" << pcd_tgt->size() << endl;
}

void staticOutlierRemoval(pcl::PointCloud<pcl::PointXYZ>::Ptr incloud, pcl::PointCloud<pcl::PointXYZ>::Ptr outcloud) {
	pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
	sor.setInputCloud(incloud);
	sor.setMeanK(5); //设置在进行统计时考虑查询点临近点数
	sor.setStddevMulThresh(2);// 设置判断是否为离群点的阈值，里面的数字表示标准差的倍数，1个标准差以上就是离群点
	//即当判断点的k紧邻平均距离（mean distance）大于全局的1倍标准差+平均距离（global distances mean and standard），则为离群点
	sor.filter(*outcloud);
	cout << "cloud filtering:\n" << outcloud->size() << endl;
}


void normal_angle_samping(pcl::PointCloud<pcl::PointXYZ>::Ptr  cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr  cloud1) {
	//--------------计算每一个点的法向量----------------
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
	n.setInputCloud(cloud);
	//设置邻域点搜索方式
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	n.setSearchMethod(tree);
	//设置KD树搜索半径
	// n.setRadiusSearch (0.03);
	n.setKSearch(10);
	//定义一个新的点云储存含有法线的值
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	//计算出来法线的值
	n.compute(*normals);

	float Angle = 0.0;
	float Average_Sum_AngleK = 0.0;//定义邻域内K个点法向量夹角的平均值
	vector<int>indexes;

	float threshold = 5;//夹角阈值**************************

	//--------------计算法向量夹角及夹角均值----------------
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;  //建立kdtree对象
	kdtree.setInputCloud(cloud); //设置需要建立kdtree的点云指针
	int K = 20;  //设置需要查找的近邻点个数
	vector<int> pointIdxNKNSearch(K);  //保存每个近邻点的索引
	vector<float> pointNKNSquaredDistance(K); //保存每个近邻点与查找点之间的欧式距离平方
	pcl::PointXYZ searchPoint;
	for (size_t i = 0; i < cloud->points.size(); ++i) {
		searchPoint = cloud->points[i];
		if (kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
		{
			float Sum_AngleK = 0.0;//定义K个邻近的点法向夹角之和
			//cout << "Neighbors within voxel search at (" << searchPoint << endl;
			for (size_t m = 0; m < pointIdxNKNSearch.size(); ++m) {
				/*计算法向量的夹角*/
				Eigen::Vector3f v1(normals->points[i].data_n[0],
					normals->points[i].data_n[1],
					normals->points[i].data_n[2]),

					v2(normals->points[pointIdxNKNSearch[m]].data_n[0],
						normals->points[pointIdxNKNSearch[m]].data_n[1],
						normals->points[pointIdxNKNSearch[m]].data_n[2]);
				//计算夹角（方法一，直接调用库函数）
				Angle = pcl::getAngle3D(v1, v2, true);
				//（方法二,自己写）
				//double radian_angle = atan2(v1.cross(v2).norm(), v1.transpose() * v2); //[0,PI]
				//Angle = radian_angle;
			}
			Sum_AngleK += Angle;//邻域夹角之和
			Average_Sum_AngleK = Sum_AngleK / pointIdxNKNSearch.size();//邻域夹角均值
	//-----------------提取特征点--------------------
			if (Average_Sum_AngleK > threshold) {
				indexes.push_back(i);
			}
		}
	}

	pcl::copyPointCloud(*cloud, indexes, *cloud1);
	cout << "提取的特征点个数:" << cloud1->points.size() << endl;
}


//num输入曲率计算后最大num点云数量
void PrincipalCurval_sampling(pcl::PointCloud<pcl::PointXYZ>::Ptr incloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr outcloud,
	int num) {

	//---------------计算点云法线--------------------
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> nor;
	nor.setInputCloud(incloud);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_nor(new pcl::search::KdTree<pcl::PointXYZ>);
	nor.setSearchMethod(tree_nor);
	nor.setKSearch(10);
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	nor.compute(*normals);

	//---------------主曲率计算--------------------------
	pcl::PrincipalCurvaturesEstimation<pcl::PointXYZ, pcl::Normal, pcl::PrincipalCurvatures> Pri;
	Pri.setInputCloud(incloud);
	Pri.setSearchMethod(tree_nor);
	Pri.setInputNormals(normals);
	Pri.setKSearch(10);
	//---------------计算主曲率-----------------------
	pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr Pri_cur(new pcl::PointCloud<pcl::PrincipalCurvatures>());
	Pri.compute(*Pri_cur);

	cout << "output points.size: " << Pri_cur->points.size() << endl;
	//获取曲率集
	vector<PCURVATURE> tempCV;
	
	float curvature = 0.0;
	PCURVATURE pv;
	for (int i = 0; i < Pri_cur->size(); i++)
	{
		//平均曲率
//		curvature = ((*Pri_cur)[i].pc1 + (*Pri_cur)[i].pc2) / 2;
		//高斯曲率
		curvature = (*Pri_cur)[i].pc1 * (*Pri_cur)[i].pc2;

		pv.index = i;
		pv.curvature = curvature;
		tempCV.insert(tempCV.end(), pv);
	}


	//获取所有曲率中的最大，最小曲率
	//选择排序
	//找到最大的前num个曲率点
	PCURVATURE temp;
	int maxIndex = 0;
	int count = 0;

	for (int i = 0; i < tempCV.size(); i++)
	{
		float maxCurvature = -99999;
		for (int j = i + 1; j < tempCV.size(); j++)
		{
			if (maxCurvature < tempCV[j].curvature) {
				maxCurvature = tempCV[j].curvature;
				maxIndex = j;
			}
		}
		if (maxCurvature>tempCV[i].curvature)
		{
			temp = tempCV[maxIndex];
			tempCV[maxIndex] = tempCV[i];
			tempCV[i] = temp;
			count++;

		}
		if (count>num)
		{
			break;
		}
	}

//	pcl::PointIndices index_1;
//	pcl::IndicesPtr  index_ptr = boost::make_shared<std::vector<int>>(pv.index);
//	boost::shared_ptr<std::vector<int>> index_ptr = boost::make_shared<std::vector<int>>(pv.index);
	pcl::PointIndices indices_0;
	for (int i = 0; i < num; i++)
	{
		indices_0.indices.push_back(tempCV[i].index);
	}
	pcl::copyPointCloud(*incloud, indices_0, *outcloud);

}

void region_Growing_diff_vis(pcl::PointCloud<pcl::PointXYZ>::Ptr  incloud_icp,
	pcl::PointCloud<pcl::PointXYZ>::Ptr  cloud_tgt) {


	//----------------------------获得两个空间对齐点云之间的差值------------------------------
	pcl::SegmentDifferences<pcl::PointXYZ> sd;  // 创建SegmentDifferences实例化分割对象
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_different(new pcl::PointCloud<pcl::PointXYZ>); // 用来存储两片点云之间的差异
	sd.setSearchMethod(tree);         // 搜索方式
	/////////////////边缘缺口和裂纹0.00016，//////////////
	sd.setDistanceThreshold(0.3);  // 设置两个输入数据集中对应点之间的最大距离公差(平方)。
	sd.setInputCloud(incloud_icp);         // 加载点云cloudA
	sd.setTargetCloud(cloud_tgt);        // 加载与点云cloudA进行比较的点云cloudB
	sd.segment(*cloud_different);     // 得到两组点云之间的差异
	cout << "差异部分有：" << cloud_different->points.size() << "个点" << endl;


	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_diff(new pcl::visualization::PCLVisualizer("Different cloud"));
	viewer_diff->setWindowName("可视化点云之间的差异");
	viewer_diff->addText("viewer the difference of cloudA and cloudB", 50, 50, 1, 0, 0, "v3_text");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> diff_color(cloud_different, 0, 0, 255);

	viewer_diff->addPointCloud<pcl::PointXYZ>(cloud_different, diff_color, "cloud difference");// 可视化差异点云

	pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::search::Search<pcl::PointXYZ>::Ptr tree_reg(new pcl::search::KdTree<pcl::PointXYZ>);
	//法线计算
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> nor;
	nor.setInputCloud(cloud_different);
	nor.setSearchMethod(tree_reg);
	nor.setKSearch(4);
	nor.compute(*normals);

	reg.setMinClusterSize(10);
	reg.setMaxClusterSize(500);
	reg.setSearchMethod(tree_reg);
	reg.setInputCloud(cloud_different);
	reg.setInputNormals(normals);
	std::vector <pcl::PointIndices> clusters;
	reg.extract(clusters);

	std::cout << "区域生长的聚类个数为：" << clusters.size() << endl;
	for (size_t i = 0; i < clusters.size(); i++)
	{
		std::cout << "聚类的里面的点云个数：" << clusters[i].indices.size() << endl;
	}

	int begin = 0;
	for (std::vector<pcl::PointIndices>::const_iterator it = clusters.begin(); it != clusters.end(); ++it)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
		for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); pit++)
		{
			cloud_cluster->points.push_back(cloud_different->points[*pit]);
		}
		cloud_cluster->width = cloud_cluster->points.size();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;

		std::stringstream ss;
		ss << ".//data//" << "diff_reg" << begin + 1 << ".pcd";
		pcl::io::savePCDFile(ss.str(), *cloud_cluster);
		cout << ss.str() << "  saved  " << endl;
		begin++;

	}

	//--------------------------------------包围盒-------------------------------------------

	//------------创建pcl::MomentOfInertiaEstimation 类-------------
	/**/
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_diff1(new pcl::PointCloud<pcl::PointXYZ>);

	vector <float> moment_of_inertia;//存储惯性矩的特征向量
	vector <float> eccentricity;//存储偏心率的特征向量
	//声明存储描述符和边框所需的所有必要变量。
	pcl::PointXYZ min_point_AABB;
	pcl::PointXYZ max_point_AABB;
	pcl::PointXYZ min_point_OBB;
	pcl::PointXYZ max_point_OBB;
	pcl::PointXYZ position_OBB;
	Eigen::Matrix3f rotational_matrix_OBB;
	float major_value, middle_value, minor_value;
	Eigen::Vector3f major_vector, middle_vector, minor_vector;
	Eigen::Vector3f mass_center;

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("基于惯性矩与偏心率的描述子"));

	if ((pcl::io::loadPCDFile("./data/diff_reg1.pcd", *cloud_diff1) != -1))
	{
		pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;
		feature_extractor.setInputCloud(cloud_diff1);
		feature_extractor.compute();

		feature_extractor.getMomentOfInertia(moment_of_inertia);//惯性矩特征
		feature_extractor.getEccentricity(eccentricity);//偏心率特征
		feature_extractor.getAABB(min_point_AABB, max_point_AABB);//AABB对应的左下角和右上角坐标
		feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);//OBB对应的相关参数
		feature_extractor.getEigenValues(major_value, middle_value, minor_value);//三个特征值
		feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);//三个特征向量
		feature_extractor.getMassCenter(mass_center);//点云中心坐标
	}

	//------------------------可视化---------------------------------
	cout << "蓝色的包围盒为OBB,红色的包围盒为AABB" << endl;

	viewer->setBackgroundColor(0, 0, 0);
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();
	viewer->addPointCloud<pcl::PointXYZ>(cloud_tgt, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_tgt, 0, 255, 0), "sample cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");
	viewer->addCube(min_point_AABB.x, max_point_AABB.x, min_point_AABB.y, max_point_AABB.y, min_point_AABB.z, max_point_AABB.z, 1.0, 0.0, 0.0, "AABB");
	viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.5, "AABB");
	viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "AABB");
	Eigen::Vector3f position(position_OBB.x, position_OBB.y, position_OBB.z);

	cout << "position_OBB: " << position_OBB << endl;
	cout << "mass_center: " << mass_center << endl;
	Eigen::Quaternionf quat(rotational_matrix_OBB);
	viewer->addCube(position, quat, max_point_OBB.x - min_point_OBB.x, max_point_OBB.y - min_point_OBB.y, max_point_OBB.z - min_point_OBB.z, "OBB");
	viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 1, "OBB");
	viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.5, "OBB");
	viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "OBB");
	viewer->setRepresentationToWireframeForAllActors();
	pcl::PointXYZ center(mass_center(0), mass_center(1), mass_center(2));
	pcl::PointXYZ x_axis(major_vector(0) + mass_center(0), major_vector(1) + mass_center(1), major_vector(2) + mass_center(2));
	pcl::PointXYZ y_axis(middle_vector(0) + mass_center(0), middle_vector(1) + mass_center(1), middle_vector(2) + mass_center(2));
	pcl::PointXYZ z_axis(minor_vector(0) + mass_center(0), minor_vector(1) + mass_center(1), minor_vector(2) + mass_center(2));
	viewer->addLine(center, x_axis, 1.0f, 0.0f, 0.0f, "major eigen vector");
	viewer->addLine(center, y_axis, 0.0f, 1.0f, 0.0f, "middle eigen vector");
	viewer->addLine(center, z_axis, 0.0f, 0.0f, 1.0f, "minor eigen vector");
	cout << "size of cloud :" << cloud_diff1->points.size() << endl;
	cout << "moment_of_inertia :" << moment_of_inertia.size() << endl;
	cout << "eccentricity :" << eccentricity.size() << endl;

	//--------------------------------------包围盒2-------------------------------------------

	//------------创建pcl::MomentOfInertiaEstimation 类-------------
	/**/
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_diff2(new pcl::PointCloud<pcl::PointXYZ>);

	vector <float> moment_of_inertia2;//存储惯性矩的特征向量
	vector <float> eccentricity2;//存储偏心率的特征向量
	//声明存储描述符和边框所需的所有必要变量。
	pcl::PointXYZ min_point_AABB2;
	pcl::PointXYZ max_point_AABB2;
	pcl::PointXYZ min_point_OBB2;
	pcl::PointXYZ max_point_OBB2;
	pcl::PointXYZ position_OBB2;
	Eigen::Matrix3f rotational_matrix_OBB2;
	float major_value2, middle_value2, minor_value2;
	Eigen::Vector3f major_vector2, middle_vector2, minor_vector2;
	Eigen::Vector3f mass_center2;

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer2(new pcl::visualization::PCLVisualizer("基于惯性矩与偏心率的描述子"));

	if ((pcl::io::loadPCDFile("./data/diff_reg2.pcd", *cloud_diff2) != -1))
	{
		pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor2;
		feature_extractor2.setInputCloud(cloud_diff2);
		feature_extractor2.compute();

		feature_extractor2.getMomentOfInertia(moment_of_inertia2);//惯性矩特征
		feature_extractor2.getEccentricity(eccentricity2);//偏心率特征
		feature_extractor2.getAABB(min_point_AABB2, max_point_AABB2);//AABB对应的左下角和右上角坐标
		feature_extractor2.getOBB(min_point_OBB2, max_point_OBB2, position_OBB2, rotational_matrix_OBB2);//OBB对应的相关参数
		feature_extractor2.getEigenValues(major_value2, middle_value2, minor_value2);//三个特征值
		feature_extractor2.getEigenVectors(major_vector2, middle_vector2, minor_vector2);//三个特征向量
		feature_extractor2.getMassCenter(mass_center2);//点云中心坐标
	}

	//------------------------可视化---------------------------------
	cout << "蓝色的包围盒为OBB,红色的包围盒为AABB" << endl;

	viewer2->setBackgroundColor(0, 0, 0);
	viewer2->addCoordinateSystem(1.0);
	viewer2->initCameraParameters();
	viewer2->addPointCloud<pcl::PointXYZ>(cloud_tgt, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_tgt, 0, 255, 0), "sample cloud");
	viewer2->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");
	viewer2->addCube(min_point_AABB2.x, max_point_AABB2.x, min_point_AABB2.y, max_point_AABB2.y, min_point_AABB2.z, max_point_AABB2.z, 1.0, 0.0, 0.0, "AABB");
	viewer2->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.5, "AABB");
	viewer2->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "AABB");
	Eigen::Vector3f position2(position_OBB2.x, position_OBB2.y, position_OBB2.z);

	cout << "position_OBB: " << position_OBB2 << endl;
	cout << "mass_center: " << mass_center2 << endl;
	Eigen::Quaternionf quat2(rotational_matrix_OBB2);
	viewer2->addCube(position2, quat2, max_point_OBB2.x - min_point_OBB2.x, max_point_OBB2.y - min_point_OBB2.y, max_point_OBB2.z - min_point_OBB2.z, "OBB");
	viewer2->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 1, "OBB");
	viewer2->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.5, "OBB");
	viewer2->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "OBB");
	viewer2->setRepresentationToWireframeForAllActors();
	pcl::PointXYZ center2(mass_center(0), mass_center(1), mass_center(2));
	pcl::PointXYZ x_axis2(major_vector(0) + mass_center(0), major_vector(1) + mass_center(1), major_vector(2) + mass_center(2));
	pcl::PointXYZ y_axis2(middle_vector(0) + mass_center(0), middle_vector(1) + mass_center(1), middle_vector(2) + mass_center(2));
	pcl::PointXYZ z_axis2(minor_vector(0) + mass_center(0), minor_vector(1) + mass_center(1), minor_vector(2) + mass_center(2));
	viewer2->addLine(center2, x_axis2, 1.0f, 0.0f, 0.0f, "major eigen vector");
	viewer2->addLine(center2, y_axis2, 0.0f, 1.0f, 0.0f, "middle eigen vector");
	viewer2->addLine(center2, z_axis2, 0.0f, 0.0f, 1.0f, "minor eigen vector");
	cout << "size of cloud :" << cloud_diff2->points.size() << endl;
	cout << "moment_of_inertia :" << moment_of_inertia2.size() << endl;
	cout << "eccentricity :" << eccentricity2.size() << endl;

	while (!viewer->wasStopped())
	{
		viewer_diff->spinOnce(100);
		viewer->spinOnce(100);
		viewer2->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}


/*计算RMSE*/
double computeCloudRMSE(pcl::PointCloud<pcl::PointXYZ>::ConstPtr target,
	pcl::PointCloud<pcl::PointXYZ>::ConstPtr source,
	double max_range)
{
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud(target);
	double fitness_score = 0.0;
	std::vector<int> nn_indices(1);
	std::vector<float> nn_dists(1);
	int nr = 0;

	for (size_t i = 0; i < source->points.size(); ++i) {
		if (!pcl_isfinite((*source)[i].x))
			continue;

		tree->nearestKSearch(source->points[i], 1, nn_indices, nn_dists);

		if (nn_dists[0] <= max_range * max_range) {
			fitness_score += nn_dists[0];
			nr++;
		}
	}

	if (nr > 0)
		return sqrt(fitness_score / nr);
	else
		return (std::numeric_limits<double>::max());
}



/*计算点云密度

输入点云

*/
double
compute_cloud_resolution(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud)
{
	double resolution = 0.0;
	int numberOfPoints = 0;
	int nres;
	std::vector<int> indices(2);
	std::vector<float> squaredDistances(2);
	pcl::search::KdTree<pcl::PointXYZ> tree;
	tree.setInputCloud(cloud);

	for (size_t i = 0; i < cloud->size(); ++i)
	{    //检查是否存在无效点
		if (!pcl::isFinite(cloud->points[i]))
			continue;

		//Considering the second neighbor since the first is the point itself.
		//在同一个点云内进行k近邻搜索时，k=1的点为查询点本身。
		nres = tree.nearestKSearch(i, 2, indices, squaredDistances);
		if (nres == 2)
		{
			resolution += sqrt(squaredDistances[1]);
			++numberOfPoints;
		}
	}
	if (numberOfPoints != 0)
		resolution /= numberOfPoints;

	return resolution;
}


bool next_iteration = false;

void keyBoardEventOccured(const pcl::visualization::KeyboardEvent& event,
	void* nothing)
{
	if (event.getKeySym() == "space" && event.keyDown()) {
		next_iteration = true;
	}
}

void  VisualizeRegistration(PointCloud::Ptr& source, PointCloud::Ptr& target, PointCloud::Ptr& icp, PointCloud::Ptr target_key, PointCloud::Ptr source_key)
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
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color_point(target_key, 255, 255, 0);
	viewer_iss->addPointCloud(target_key, target_color_point, "key_target cloud", v3);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_color_point(source_key, 255, 0, 255);
	viewer_iss->addPointCloud(source_key, source_color_point, "key_source cloud", v4);

	//添加坐标系
//	viewer->addCoordinateSystem(10);
//	viewer_iss->addCoordinateSystem(10);
	//viewer->initCameraParameters();
	while ((!viewer->wasStopped()) && (!viewer_iss->wasStopped()))
	{
		viewer_iss->spinOnce(100);
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(10000));
	}

}



//不共线的三点确定一个平面
inline Eigen::Vector4d cal_plane(const pcl::PointXYZ& a, const pcl::PointXYZ& b, const pcl::PointXYZ& c) {

	Eigen::Vector4d parm;
	Eigen::Vector3d p1, p2, p3, p1p2, p1p3, N, N1;
	p1 << a.x, a.y, a.z;
	p2 << b.x, b.y, b.z;
	p3 << c.x, c.y, c.z;
	p1p2 = p2 - p1;
	p1p3 = p3 - p1;

	N = p1p2.cross(p1p3); //平面法向量
	N1 = N / N.norm();    //平面的单位法向量

	parm[0] = N1[0];
	parm[1] = N1[1];
	parm[2] = N1[2];
	parm[3] = -N1.dot(p1);

	return parm;

}

// 用于将参数传递给回调函数的结构体

struct CallbackArgs {
	pcl::PointCloud<pcl::PointXYZ>::Ptr clicked_points_3d;
	pcl::visualization::PCLVisualizer::Ptr viewerPtr;
};


void pickPointCallback(const pcl::visualization::PointPickingEvent &event, void *args
) {
	CallbackArgs *data = (CallbackArgs *)args;
	if (event.getPointIndex() == -1)
		return;
	PointT current_point;
	event.getPoint(current_point.x, current_point.y, current_point.z);
	data->clicked_points_3d->points.push_back(current_point);
	float point_to_point_dis = 0;
	float point_to_plane_dis = 0;

	for (int i = 0; i < point_data_all_vec.size(); i++)
	{
		//		cout << point_data_all_vec[i].point_data.x;
		if ((current_point.x == point_data_all_vec[i].point_data.x) && (current_point.y == point_data_all_vec[i].point_data.y) && (current_point.z == point_data_all_vec[i].point_data.z))
		{
			cout << "当前对应点的距离为： " << point_data_all_vec[i].rmse_point << endl;
			cout << "当前对应点的到拟合平面的距离为： " << point_data_all_vec[i].pointTodistance_1 << endl;
			point_to_point_dis = point_data_all_vec[i].rmse_point;
			point_to_plane_dis = point_data_all_vec[i].pointTodistance_1;
		}
	}

	for (int i = 0; i < point_data_all_vec_1.size(); i++)
	{
		//		cout << point_data_all_vec[i].point_data.x;
		if ((current_point.x == point_data_all_vec_1[i].point_data.x) && (current_point.y == point_data_all_vec_1[i].point_data.y) && (current_point.z == point_data_all_vec_1[i].point_data.z))
		{
			cout << "当前对应点的距离为： " << point_data_all_vec_1[i].rmse_point << endl;
			cout << "当前对应点的到拟合平面的距离为： " << point_data_all_vec_1[i].pointTodistance_1 << endl;
			point_to_point_dis = point_data_all_vec_1[i].rmse_point;
			point_to_plane_dis = point_data_all_vec_1[i].pointTodistance_1;
			
		}
	}


	// 绘制红色点
	pcl::visualization::PointCloudColorHandlerCustom<PointT> red(data->clicked_points_3d, 255, 0, 0);
	data->viewerPtr->removePointCloud("clicked_points");
	data->viewerPtr->addPointCloud(data->clicked_points_3d, red, "clicked_points");
	data->viewerPtr->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10,
		"clicked_points");
	std::cout << "当前获取的点的位置："<<" x = " << current_point.x << " y = " << current_point.y << " z = " << current_point.z << std::endl;
	std::stringstream str;
	
	str.str("");
	str <<"now click the point = " << current_point.x << " " << current_point.y << " " << current_point.z << endl
		<<"point to point distance: " << point_to_point_dis << endl 
		<<"point to plane distance: " << point_to_plane_dis << endl;

 
	bool bool_flag = data->viewerPtr->updateText(str.str(), 10, 20, 16, 1, 1, 1, "clicked_points_every"); //text不能是中文
	cout << bool_flag;
	std::stringstream str1;
}


//获取点的量化指标--------------对应点的rmse，source和target的欧式距离的最大值，最小值，和均平方根误差。
void pointcloud_Quantitative_Analysis(pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud,
	float R_ktree) {

	//初始化对象
	//建立kd-tree  寻找source的对应点及rmse
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_target;
	kdtree_target.setInputCloud(target_cloud);
	float dist;//点到平面的距离

	std::vector<float> _source_sqrt_pointRadiusSquareDistance1;//保存近邻点与查找点之间的欧式距离
	std::vector<float> _source_sqrt_pointTodistance;//保存近邻点与查找点之间的欧式距离
	std::vector<int> source_pointIdxRadiusSearch1; //保存近邻点的索引

	for (size_t i = 0; i < source_cloud->size(); i++)
	{
		pcl::PointXYZ searchPoint = source_cloud->points[i]; //设置查找点
		std::vector<int> pointIdxRadiusSearch; //保存每个近邻点的索引
		std::vector<float> pointRadiusSquareDistance;//保存每个近邻点与查找点之间的欧式距离平方
		point_data_all_1.index = i;

		point_data_all_1.point_data = searchPoint;

		float radius = 4 * R_ktree;//设置查找半径范围
		//k近邻查找对应点
		if (kdtree_target.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquareDistance, 1) > 0)
		{
			for (size_t j = 0; j < pointIdxRadiusSearch.size(); j++)
			{
//				if (pointRadiusSquareDistance[j] > 0)
//				{
					//cout << "(distance:" << sqrt(pointRadiusSquareDistance[j]) * 100 << ")" << endl;
					_source_sqrt_pointRadiusSquareDistance1.push_back(sqrt(pointRadiusSquareDistance[j]));
					source_pointIdxRadiusSearch1.push_back(pointIdxRadiusSearch[j]);
					point_data_all_1.rmse_point = sqrt(pointRadiusSquareDistance[j]);
//				}
			}

		}
		//k近邻查找对应点中的三个点
		if (kdtree_target.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquareDistance, 3) > 0)
		{
			for (size_t j = 0; j < pointIdxRadiusSearch.size()-2; j++)
			{
//				cout << " " << target_cloud->points[pointRadiusSquareDistance[j]];

				//从点云选取三个点获取切片平面
				pcl::PointXYZ pa = target_cloud->points[pointIdxRadiusSearch[j]];
				pcl::PointXYZ pb = target_cloud->points[pointIdxRadiusSearch[j+1]];
				pcl::PointXYZ pc = target_cloud->points[pointIdxRadiusSearch[j+2]];
				Eigen::Vector4d n;  //平面方程系数
				n = cal_plane(pa, pb, pc);

				auto dis = pcl::pointToPlaneDistance(searchPoint, n[0], n[1], n[2], n[3]); //点到平面的距离
//				cout << "点到平面的距离为： " << dis << endl;

				_source_sqrt_pointTodistance.push_back(dis);
				point_data_all_1.pointTodistance_1 = dis;
			}

			

		}
		point_data_all_vec_1.insert(point_data_all_vec_1.end(), point_data_all_1);
		//		cout << "R=4 * source_resolution_source_sampling近邻点个数： " << pointIdxRadiusSearch.size() << endl;
	}

	std::vector<float>::iterator biggest1 = std::max_element(std::begin(_source_sqrt_pointRadiusSquareDistance1), std::end(_source_sqrt_pointRadiusSquareDistance1));
	auto smallest1 = std::min_element(std::begin(_source_sqrt_pointRadiusSquareDistance1), std::end(_source_sqrt_pointRadiusSquareDistance1));
	//	std::cout << "min element is " << *smallest << " at position " << std::distance(std::begin(sqrt_pointRadiusSquareDistance), smallest) << std::endl;

	float sum1 = 0.0, rmse1;

	for (size_t i = 0; i < _source_sqrt_pointRadiusSquareDistance1.size(); i++)
	{
		sum1 += _source_sqrt_pointRadiusSquareDistance1[i];

	}

	rmse1 = sqrt(sum1 / _source_sqrt_pointRadiusSquareDistance1.size());
	//	cout << "匹配点对个数" << all_correspondences.size() << endl;
	cout << "source_pointcloud的对应点距离最大值biggest1:  " << *biggest1 << "毫米mm" << endl;
	cout << "source_pointcloud的对应点距离最小值smallest1:  " << *smallest1 << "毫米mm" << endl;
	cout << "source_pointcloud的对应点均方根误差rmse1:  " << rmse1 << "毫米mm" << endl;



	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_source;
	kdtree_source.setInputCloud(source_cloud);

	std::vector<float> _source_sqrt_pointRadiusSquareDistance;//保存近邻点与查找点之间的欧式距离
	std::vector<int> source_pointIdxRadiusSearch; //保存近邻点的索引
	float dist1;//点到平面的距离

	for (size_t i = 0; i < target_cloud->size(); i++)
	{
		pcl::PointXYZ searchPoint = target_cloud->points[i]; //设置查找点
		std::vector<int> pointIdxRadiusSearch; //保存每个近邻点的索引
		std::vector<float> pointRadiusSquareDistance;//保存每个近邻点与查找点之间的欧式距离平方
		point_data_all.index = i;

		point_data_all.point_data = searchPoint;

		float radius = 4 * R_ktree;//设置查找半径范围
		//k近邻查找对应点最近的1个点
		if (kdtree_source.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquareDistance, 1) > 0)
		{
			for (size_t j = 0; j < pointIdxRadiusSearch.size(); j++)
			{
//				if (pointRadiusSquareDistance[j] >= 0)
//				{
					//cout << "(distance:" << sqrt(pointRadiusSquareDistance[j]) * 100 << ")" << endl;
					_source_sqrt_pointRadiusSquareDistance.push_back(sqrt(pointRadiusSquareDistance[j]));
					source_pointIdxRadiusSearch.push_back(pointIdxRadiusSearch[j]);
					point_data_all.rmse_point = sqrt(pointRadiusSquareDistance[j]);

//				}
			}

		}

		//k近邻查找对应点最近的三个点
		if (kdtree_source.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquareDistance, 3) > 0)
		{
			for (size_t j = 0; j < pointIdxRadiusSearch.size() - 2; j++)
			{
				//从点云选取三个点获取切片平面
				pcl::PointXYZ pa = source_cloud->points[pointIdxRadiusSearch[j]];
				pcl::PointXYZ pb = source_cloud->points[pointIdxRadiusSearch[j + 1]];
				pcl::PointXYZ pc = source_cloud->points[pointIdxRadiusSearch[j + 2]];
				Eigen::Vector4d n;  //平面方程系数
				n = cal_plane(pa, pb, pc);

				auto dis = pcl::pointToPlaneDistance(searchPoint, n[0], n[1], n[2], n[3]);//点到平面的距离
				point_data_all.pointTodistance_1 = dis;
			}

		}
		//按照
		point_data_all_vec.insert(point_data_all_vec.end(), point_data_all);
		//cout << "R=4 * source_resolution_source_sampling近邻点个数： " << pointIdxRadiusSearch.size() << endl;
	}
	std::vector<float>::iterator biggest = std::max_element(std::begin(_source_sqrt_pointRadiusSquareDistance), std::end(_source_sqrt_pointRadiusSquareDistance));

	auto smallest = std::min_element(std::begin(_source_sqrt_pointRadiusSquareDistance), std::end(_source_sqrt_pointRadiusSquareDistance));
	//	std::cout << "min element is " << *smallest << " at position " << std::distance(std::begin(sqrt_pointRadiusSquareDistance), smallest) << std::endl;

	float sum = 0.0, rmse;

	for (size_t i = 0; i < _source_sqrt_pointRadiusSquareDistance.size(); i++)
	{
		sum += _source_sqrt_pointRadiusSquareDistance[i];

	}

	rmse = sqrt(sum / _source_sqrt_pointRadiusSquareDistance.size());
	//	cout << "匹配点对个数" << all_correspondences.size() << endl;
	cout << "target_pointlcoud的对应点距离最大值biggest:  " << *biggest  << "毫米mm" << endl;
	cout << "target_pointlcoud的对应点距离最小值smallest:  " << *smallest  << "毫米mm" << endl;
	cout << "target_pointlcoud的对应点均方根误差rmse:  " << rmse  << "毫米mm" << endl;

	//计算第0个点到所有点的距离，映射到0到255之间  距离越远越红
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
	for (size_t i = 0; i < _source_sqrt_pointRadiusSquareDistance1.size(); i++)
	{
		pcl::PointXYZRGB point;
		point.x = source_cloud->points[i].x;
		point.y = source_cloud->points[i].y;
		point.z = source_cloud->points[i].z;

		float dist = _source_sqrt_pointRadiusSquareDistance1[i] * 100;
		if (i % 100 == 0)
			cout << "dist: " << dist << endl;

		uint8_t r, g, b;
		if (dist > 40) {
			r = 255;
		}
			
		else {
			r = int(dist) * 5;
			
		}
		b = 255 - r;
		//g = 255 - r * 2;
		if (r > 125)
			//b = g;
			g = b;
		else
		{
			g = r;
			//b = r;
		}
		uint32_t rgba = (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
		float rgbf = *reinterpret_cast<float*>(&rgba);
		point.rgb = rgbf;
		cloud_rgb->push_back(point);
	}
	//连接xyz和对应点欧式距离
//	pcl::PointCloud<float>::Ptr target_cloud_dis(new pcl::PointCloud<float>);
//	pcl::concatenateFields(*target_cloud, _source_sqrt_pointRadiusSquareDistance, *source_cloud);
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_rmse(new pcl::visualization::PCLVisualizer("Cloud Viewer"));//直接创造一个显示窗口
	viewer_rmse->addPointCloud<pcl::PointXYZRGB>(cloud_rgb, "cloud_rgb cloud");//再这个窗口显示点云

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Simple Cloud Viewer"));//直接创造一个显示窗口
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_color(source_cloud, 255, 255, 0);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color(target_cloud, 255, 0, 255);

	viewer->addPointCloud<pcl::PointXYZ>(source_cloud, source_color, "source cloud");//再这个窗口显示点云
	viewer->addPointCloud<pcl::PointXYZ>(target_cloud, target_color, "target cloud");
	viewer->addText("click_point", 10, 30, 16, 1, 1, 1, "clicked_points_every");
//	viewer->addText("click_point_dis", 10, 10, 16, 1, 1, 1, "clicked_points_dis");

	// 添加点拾取回调函数
	CallbackArgs  cb_args;
	pcl::PointCloud<pcl::PointXYZ>::Ptr clicked_points_3d(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_click(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr target_click(new pcl::PointCloud<pcl::PointXYZ>);
	cb_args.clicked_points_3d = clicked_points_3d;
	//	cb_args.point_data_all_vec = point_data_all_vec;
	cb_args.viewerPtr = pcl::visualization::PCLVisualizer::Ptr(viewer);
	viewer->registerPointPickingCallback(pickPointCallback, (void*)&cb_args);
	std::cout << "Shift+click on three floor points, then press 'Q'..." << std::endl;

	viewer->addCoordinateSystem(10);

	while (!viewer->wasStopped())
	{
		viewer_rmse->spinOnce(100);
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}

}





void statical_removel(pcl::PointCloud<pcl::PointXYZ>::Ptr incloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr outcloud,
	int set_k) {

	pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
	sor.setInputCloud(incloud);
	sor.setMeanK(set_k);  //设置在进行统计时考虑查询点临近点数
	sor.setStddevMulThresh(4);  // 设置判断是否为离群点的阈值，里面的数字表示标准差的倍数，1个标准差以上就是离群点。
	//即：当判断点的k近邻平均距离（mean distance）大于全局1倍标准差+平均距离（global distance mean and standard）,则为离群点
	sor.filter(*outcloud);
	cout << "cloud after filtering:\n" << outcloud->size() << endl;

}




//获取配准后的点云切片
void get_slice_pointcloud(
	pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud,
	float thickness,
	pcl::PointCloud<pcl::PointXYZ>::Ptr slice_points_cloud
) {
	//点云切面
	//pcl::PointCloud<pcl::PointXYZ>::Ptr slice_points_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	//slice2(cloud_source, slice_points_cloud, 5);

	//------------------------------- 获取点云最值 ---------------------------------
	PointT min, max;
	pcl::getMinMax3D(*in_cloud, min, max);

	cout << "->min_x = " << min.x << endl;
	cout << "->min_y = " << min.y << endl;
	cout << "->min_z = " << min.z << endl;
	cout << "->max_x = " << max.x << endl;
	cout << "->max_y = " << max.y << endl;
	cout << "->max_z = " << max.z << endl;

	float a = 1;   //pa x
	float b;   //pb y     竖
	float c = 1;   //pc z
	float diff = max.y - min.y;

	b = min.y + diff / 10; //切平面位置
	//从点云选取三个点获取切片平面
	pcl::PointXYZ pa = { a,b,0 };
	pcl::PointXYZ pb = { 0,b,0 };
	pcl::PointXYZ pc = { 0,b,c };
	Eigen::Vector4d n;  //平面方程系数

	n = cal_plane(pa, pb, pc);
	cout << "平面方程系数：\n" << "a=" << n[0] << "\tb=" << n[1] << "\tc=" << n[2] << "\td=" << n[3] << endl;
	//-------------------切片提取------------------------//
	double Delta = thickness;  //设置切片的厚度


	std::vector<int> point_idex;
	for (int i = 0; i < in_cloud->size(); i++)
	{
		double Wr = n[0] * (*in_cloud)[i].x + n[1] * (*in_cloud)[i].y + n[2] * (*in_cloud)[i].z + n[3] - Delta;
		double Wl = n[0] * (*in_cloud)[i].x + n[1] * (*in_cloud)[i].y + n[2] * (*in_cloud)[i].z + n[3] + Delta;
		if (Wr * Wl <= 0)
		{
			point_idex.push_back(i);
		}
	}
	
	pcl::copyPointCloud(*in_cloud, point_idex, *slice_points_cloud);
}

//获取配准后的点云差值
void get_pointcloud_diff(
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud,
	float diff_delta
) {
	
	//----------------------------获得两个空间对齐点云之间的差值------------------------------
	pcl::SegmentDifferences<pcl::PointXYZ> sd;  // 创建SegmentDifferences实例化分割对象
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_different(new pcl::PointCloud<pcl::PointXYZ>); // 用来存储两片点云之间的差异
	sd.setSearchMethod(tree);         // 搜索方式
	/////////////////边缘缺口和裂纹0.00016，//////////////
	double shreshold = diff_delta;   // 0.3 * source_resolution_source_key;
	sd.setDistanceThreshold(shreshold);  // 设置两个输入数据集中对应点之间的最大距离公差(平方)。
	sd.setInputCloud(source_cloud);         // 加载点云cloudA
	sd.setTargetCloud(target_cloud);        // 加载与点云cloudA进行比较的点云cloudB
	sd.segment(*cloud_different);     // 得到两组点云之间的差异
	cout << "差异部分有：" << cloud_different->points.size() << "个点" << endl;

	clock_t end_diff1 = clock();

	staticOutlierRemoval(cloud_different, cloud_different);

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_diff(new pcl::visualization::PCLVisualizer("3D Viewer cloudB"));
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> transe(cloud_different, 255, 255, 0);
	//	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> transe(cloud_different, 255, 0, 0);
	viewer_diff->setWindowName("可视化缺损部分点云cloud_different");
	viewer_diff->addText("viewer of 可视化缺损部分点云cloud_different", 50, 50, "v3_text");
	viewer_diff->addPointCloud<pcl::PointXYZ>(cloud_different, transe, "cloud_different");

	while (!viewer_diff->wasStopped())
	{
		viewer_diff->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
		
}

//计算点云的FPFH
pcl::PointCloud<pcl::FPFHSignature33> compute_fpfh_feature(PointCloud::Ptr input_cloud) {
	//计算表面法线
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne_src;
	ne_src.setInputCloud(input_cloud);
	pcl::search::KdTree< pcl::PointXYZ>::Ptr tree_src(new pcl::search::KdTree< pcl::PointXYZ>());
	ne_src.setSearchMethod(tree_src);
	pcl::PointCloud<pcl::Normal>::Ptr cloud_src_normals(new pcl::PointCloud< pcl::Normal>);
	//ne_src.setRadiusSearch(0.8); //使用一个半径为2cm的球体中的所有邻居点
	ne_src.setNumberOfThreads(8);
	ne_src.setKSearch(85);
	ne_src.compute(*cloud_src_normals);// 计算特征

	//计算FPFH
	pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh_src;
	//pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh_src;
	fpfh_src.setInputCloud(input_cloud);
	fpfh_src.setInputNormals(cloud_src_normals);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_fpfh(new pcl::search::KdTree<pcl::PointXYZ>);
	fpfh_src.setSearchMethod(tree_fpfh);
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs_src(new pcl::PointCloud<pcl::FPFHSignature33>());
	fpfh_src.setKSearch(90);
	fpfh_src.setNumberOfThreads(8);
	//fpfh_src.setRadiusSearch(0.8);
	fpfh_src.compute(*fpfhs_src);

	return *fpfhs_src;

}

float computeDistance(pcl::FPFHSignature33& f1, pcl::FPFHSignature33& f2) {

	float diff = 0;
	for (size_t i = 0; i < 33; i++)
	{
		diff += (f1.histogram[i] - f2.histogram[i])*(f1.histogram[i] - f2.histogram[i]);
	}
	return diff;
}

int main(int argc, char** argv)
{
	PointCloud::Ptr cloud_target(new PointCloud), cloud_target_normal(new PointCloud);
	pcl::io::loadPLYFile("H:/pcl_study/icp_svd_demo1/icp_svd_demo1/data/边缘缺失/020-87-打磨后.ply", *cloud_target);

	// 加载点云文件
	PointCloud::Ptr cloud_source(new PointCloud), cloud_source_normal(new PointCloud);    // 源点云，待配准
	pcl::io::loadPLYFile("H:/pcl_study/icp_svd_demo1/icp_svd_demo1/data/边缘缺失/020-87-打磨前.ply", *cloud_source);

	//采样前cloud_target，cloud_source的点云密度
	float source_resolution_target = compute_cloud_resolution(cloud_target);
	float source_resolution_source = compute_cloud_resolution(cloud_source);
	cout << "采样前cloud_target点云密度为：" << source_resolution_target << endl;
	cout << "采样前cloud_source点云密度为：" << source_resolution_source << endl;

	//采样大小 
	float tgt_samping = 20 * source_resolution_target;
	float source_samping = 20 * source_resolution_target;

	float tgt_show = 5 * source_resolution_target;
	float source_show = 5 * source_resolution_target;

	//可视化所使用的源点云与目标点云
	PointCloud::Ptr cloud_target_show(new PointCloud), cloud_source_show(new PointCloud);
	downingSampling(cloud_target, cloud_target_show, tgt_show);
	downingSampling(cloud_source, cloud_source_show, source_show);

	//配准所使用的源点云与目标点云
	downingSampling(cloud_target, cloud_target, tgt_samping);
	downingSampling(cloud_source, cloud_source, source_samping);
	

	statical_removel(cloud_target, cloud_target, 50);
	statical_removel(cloud_source, cloud_source, 50);
	//采样后cloud_target，cloud_source的点云密度
	float source_resolution_target_sampling = compute_cloud_resolution(cloud_target);
	float source_resolution_source_sampling = compute_cloud_resolution(cloud_source);
	cout << "采样后cloud_target点云密度为：" << source_resolution_target_sampling << endl;
	cout << "采样后cloud_source点云密度为：" << source_resolution_source_sampling << endl;
	
	//normal_angle_samping(cloud_target, cloud_target_normal);
	//normal_angle_samping(cloud_source, cloud_source_normal);

	////融合几何采样后的cloud_target，cloud_source的点云密度
	//float source_resolution_target_normal = compute_cloud_resolution(cloud_target_normal);
	//float source_resolution_source_normal = compute_cloud_resolution(cloud_source_normal);
	//cout << "融合几何采样后的source_resolution_target_normal点云密度为：" << source_resolution_target_normal << endl;
	//cout << "融合几何采样后的source_resolution_source_normal点云密度为：" << source_resolution_source_normal << endl;

	PointCloud::Ptr source_key(new PointCloud), target_key(new PointCloud);
	//曲率采样
	PrincipalCurval_sampling(cloud_target, target_key, 2300);
	PrincipalCurval_sampling(cloud_source, source_key, 2300);

	


	//--------------------------K4PCS算法进行配准------------------------------
	clock_t start_k4pcs = clock();
	cout << "目标点的关键点个数为：" << target_key->points.size() << endl;
	cout << "源点云的关键点个数为：" << source_key->points.size() << endl;
	pcl::registration::KFPCSInitialAlignment<pcl::PointXYZ, pcl::PointXYZ> kfpcs;
	kfpcs.setInputSource(source_key);  // 源点云
	kfpcs.setInputTarget(target_key);  // 目标点云
	kfpcs.setApproxOverlap(0.9);// 源和目标之间的近似重叠。 2 * source_resolution_target_sampling
	double threshold_setLambda = 1.4 * source_resolution_target_sampling;
	kfpcs.setLambda(0.4); // 平移矩阵的加权系数。(暂时不知道是干什么用的) 
	double threshold_Delta = 0.5 * source_resolution_target_sampling;
	kfpcs.setDelta(threshold_Delta, false);  // 配准后源点云和目标点云之间的距离0.6
	kfpcs.setNumberOfThreads(8);   // OpenMP多线程加速的线程数
	kfpcs.setNumberOfSamples(200); // 配准时要使用的随机采样点数量
	//kfpcs.setMaxComputationTime(1000);//最大计算时间(以秒为单位)。
	pcl::PointCloud<pcl::PointXYZ>::Ptr kpcs(new pcl::PointCloud<pcl::PointXYZ>);
	kfpcs.align(*kpcs);
	Eigen::Matrix4f k4pcs_trans;
	k4pcs_trans = kfpcs.getFinalTransformation();

	clock_t start_icp = clock();
	std::cout << "K-4PCS配准用时： " << (double)(start_icp - start_k4pcs) / (double)CLOCKS_PER_SEC << " s" << std::endl;
	cout << "变换矩阵：\n" << k4pcs_trans << endl;

	//icp配准

	pcl::PointCloud<pcl::PointXYZ>::Ptr icp_result(new pcl::PointCloud<pcl::PointXYZ>), icp_rmse(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	icp.setInputSource(source_key);
	icp.setInputTarget(target_key);
	//	icp.setMaxCorrespondenceDistance(0.1);
	icp.setMaximumIterations(100);        // 最大迭代次数50
	icp.setTransformationEpsilon(1e-10); // 两次变化矩阵之间的差值
	icp.setEuclideanFitnessEpsilon(0.001);// 均方误差
	icp.setUseReciprocalCorrespondences(true);
	icp.align(*icp_result, k4pcs_trans);
	clock_t end_icp = clock();

	//for (auto& point : *icp_result)
	//	std::cout << point << std::endl;


	PointCloud::Ptr icp_cloud_res(new PointCloud);
	pcl::transformPointCloud(*cloud_source_show, *icp_cloud_res, icp.getFinalTransformation());
	std::cout << "has converged:" << icp.hasConverged() << " score: " <<
	icp.getFitnessScore() << std::endl;    //配准分析...icp.hasConverge()=1,表示收敛
	std::cout << icp.getFinalTransformation() << std::endl;    //获取变换矩阵


	////计算FPFH
	//pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs_tgt(new pcl::PointCloud<pcl::FPFHSignature33>());
	//pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs_src(new pcl::PointCloud<pcl::FPFHSignature33>());
	//*fpfhs_src = compute_fpfh_feature(icp_cloud_res);
	//*fpfhs_tgt = compute_fpfh_feature(cloud_target);
	//clock_t end_fpfh1 = clock();

	

	pointcloud_Quantitative_Analysis(icp_cloud_res, cloud_target_show, source_resolution_source_sampling);

	////	//--------------------------可视化配准结果----------------------
//	VisualizeRegistration(cloud_source, cloud_target, icp_cloud_res, target_key, source_key);





	//Eigen::Matrix4f FinalTrans;
	////----------------------------点云配准阈值----------------------
	//int RansacIterations = 500;
	//float RansacDistance = 0.20;
	//float Fraction = 0.10;
	//float MaxDistance = 0.2;  //icp的参数
	//int MaxIterations = 200;   //icp的参数
	////-----------------------------执行配准-------------------------
	//Registration PCR(RansacIterations, RansacDistance, /*Fraction,*/ MaxDistance, MaxIterations);
	//FinalTrans = PCR.RegistrationTransform(source_key, target_key, fpfhs_tgt, fpfhs_src);

	//PointCloud::Ptr icp(new PointCloud);
	//pcl::transformPointCloud(*cloud_source, *icp, FinalTrans);


	//////////-------------误差分析---------------///////////
	//std::cout << "RMSE= " << computeCloudRMSE(cloud_target, icp, 0.1) << endl;

	////	region_Growing_diff_vis(icp, cloud_target);
	//	//--------------------------可视化配准结果----------------------
	//PCR.VisualizeRegistration(cloud_source, cloud_target, icp, target_key, source_key);



	//	//------------------迭代次数可视化----------------------//
	//	//RANSAC配准
	//	pcl::console::print_highlight("开始进行配准\n");
	//	pcl::console::print_highlight("开始进行的源配准\n");
	//	std::cout << source_key->size() << endl;
	//	pcl::console::print_highlight("开始进行目标配准\n");
	//	std::cout << target_key->size() << endl;
	//	pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> ransac;
	//	ransac.setInputSource(source_key);
	//	//	ransac.setSourceFeatures(sps_src);
	//	ransac.setSourceFeatures(fpfhs_src);
	//	ransac.setInputTarget(target_key);
	//	//	ransac.setTargetFeatures(sps_tgt);
	//	ransac.setTargetFeatures(fpfhs_tgt);
	//	ransac.setMaximumIterations(200); //  采样一致性迭代次数
	//	ransac.setNumberOfSamples(200);                    //  创建假设所需的样本数
	////	ransac.setCorrespondenceRandomness(10);          //  使用的临近特征点的数目
	//
	//	ransac.setMaxCorrespondenceDistance(0.2); // 判断是否为内点的距离阈值
	////	ransac.setInlierFraction(m_Fraction);                  //  接受位姿假设所需的内点比例
	//
	//	PointCloud::Ptr sac_result(new PointCloud);
	//	ransac.align(*sac_result);
	//
	//	Eigen::Matrix4f ransac_trans;
	//	ransac_trans = ransac.getFinalTransformation();
	//	clock_t sac_time = clock();
	//
	//
	//
	//	//icp配准
	//	PointCloud::Ptr icp_result(new PointCloud);
	//	pcl::IterativeClosestPoint<PointT, PointT> icp;
	//
	//	PointCloud::Ptr icp_cloud_iter(new PointCloud);
	//	pcl::transformPointCloud(*cloud_source, *icp_cloud_iter, ransac_trans);
	//
	//	*icp_result = *sac_result;
	//	icp.setInputSource(icp_result);
	//	icp.setInputTarget(target_key);
	//
	//	//	icp.setMaxCorrespondenceDistance(m_MaxDistance);
	//	icp.setMaximumIterations(1);        // 最大迭代次数
	//	icp.setTransformationEpsilon(1e-10); // 两次变化矩阵之间的差值
	//	icp.setEuclideanFitnessEpsilon(0.001);// 均方误差
	//
	//	//------------------迭代次数可视化及RMSE的分析----------------------//
	//	pcl::visualization::PCLVisualizer viewer_iter("RegistrationCloud_iter");
	//	int v5 = 0;
	//	int v6 = 1;
	//	viewer_iter.createViewPort(0, 0, 0.5, 1, v5);
	//	viewer_iter.createViewPort(0.5, 0, 1, 1, v6);
	//	viewer_iter.setBackgroundColor(0, 0, 0, v5);
	//	viewer_iter.setBackgroundColor(0.05, 0, 0, v6);
	//
	//	//原始点云绿色
	//	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> src_h_iter(cloud_source, 0, 255, 0);
	//	//目标点云蓝色
	//	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> tgt_h_iter(cloud_target, 0, 0, 255);
	//	//转换后的源点云红色
	//	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> transe_iter(icp_cloud_iter, 255, 0, 0);
	//	//viewer->setBackgroundColor(255, 255, 255);
	//	viewer_iter.addPointCloud(cloud_source, src_h_iter, "source cloud", v5);
	//	viewer_iter.addPointCloud(cloud_target, tgt_h_iter, "target cloud", v5);
	//
	//	viewer_iter.addPointCloud(cloud_target, tgt_h_iter, "target cloud1", v6);
	//	viewer_iter.addPointCloud(icp_cloud_iter, transe_iter, "pcs cloud", v6);
	//
	//	viewer_iter.registerKeyboardCallback(&keyBoardEventOccured, (void*)NULL);
	//
	//	int iterations = 0;    //迭代次数
	//
	//	while (!viewer_iter.wasStopped())
	//	{ // Display the visualiser until 'q' key is pressed
	//		viewer_iter.spinOnce();    //运行视图
	//		if (next_iteration)
	//		{
	//			icp.align(*icp_result, ransac_trans);    //ICP配准对齐结果
	//
	//			
	//			pcl::transformPointCloud(*icp_cloud_iter, *icp_cloud_iter, icp.getFinalTransformation());
	//
	//			std::cout << "has converged:" << icp.hasConverged() << " score: " <<
	//				icp.getFitnessScore() << std::endl;    //配准分析...icp.hasConverge()=1,表示收敛
	//			std::cout << icp.getFinalTransformation() << std::endl;    //获取变换矩阵
	//			std::cout << "Iteration = " << ++iterations; //迭代次数加1
	//			std::cout << "RMSE= " << computeCloudRMSE(target_key, icp_result, 0.1) << endl;
	//			if (iterations == 50)    //迭代满100次停止迭代
	//				return 0;
	//			viewer_iter.updatePointCloud(icp_cloud_iter, transe_iter, "aligned_cloud_v2");    //更新点云
	//		}
	//		next_iteration = false;    //本次迭代结束，不直接进行下一次迭代，等待下次迭代触发
	//	}
	//	return 0;

	return 0;
}
