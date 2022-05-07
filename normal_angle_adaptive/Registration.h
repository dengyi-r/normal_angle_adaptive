#pragma once
#pragma once
#define BOOST_TYPEOF_EMULATION   //在.cpp文件最顶部添加此语句是为了解决PCL点云处理typeof_impl.hpp报错
#include <pcl/registration/sample_consensus_prerejective.h> // RANSAC配准
#include <pcl/registration/ia_ransac.h>
#include <pcl/registration/icp.h>      // icp配准
#include <boost/thread/thread.hpp>
#include <pcl/common/common.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl\features\fpfh_omp.h>//omp加速计算
#include <pcl/features/normal_3d_omp.h>
#include <pcl/visualization/pcl_visualizer.h> // 可视化

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;

class Registration
{
	//数据成员
	int m_RansacIterations;  // RANSAC算法的最大迭代次数
	float m_RansacDistance;  // RANSAC算法判断是否为内点的距离阈值
//	float m_Fraction;        // RANSAC算法的内点比例
	float m_MaxDistance;     // ICP精配准中对应点间最大距离
	int m_MaxIterations;     // ICP最大迭代次数
public:
	Registration(int RansacIterations, float RansacDistance, /*float Fraction,*/ float MaxDistance, int MaxIterations) :
		m_RansacIterations(RansacIterations), m_RansacDistance(RansacDistance), /*m_Fraction(Fraction),*/ m_MaxDistance(MaxDistance), m_MaxIterations(MaxIterations) {}
	~Registration() {}

	// 成员函数

//	Eigen::Matrix4f RegistrationTransform(
//		PointCloud::Ptr& key_src, 
//		pcl::PointCloud<pcl::ShapeContext1980>::Ptr sps_src,
//		pcl::PointCloud<PointT>::Ptr& key_tgt, 
//		pcl::PointCloud<pcl::ShapeContext1980>::Ptr sps_tgt);

	Eigen::Matrix4f RegistrationTransform(PointCloud::Ptr & key_src, 
		PointCloud::Ptr & key_tgt, 
		pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs_tgt, 
		pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs_src);

	void VisualizeRegistration(PointCloud::Ptr & source, PointCloud::Ptr & target, PointCloud::Ptr & icp, PointCloud::Ptr target_key, PointCloud::Ptr source_key);

	void Nicp_reg(PointCloud::Ptr & key_src, PointCloud::Ptr & key_tgt, pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs_tgt, pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs_src);

	void geometry_sampling(PointCloud::Ptr & incloud, PointCloud::Ptr & outcloud);


};
