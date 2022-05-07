#pragma once
#pragma once
#define BOOST_TYPEOF_EMULATION   //��.cpp�ļ������Ӵ������Ϊ�˽��PCL���ƴ���typeof_impl.hpp����
#include <pcl/registration/sample_consensus_prerejective.h> // RANSAC��׼
#include <pcl/registration/ia_ransac.h>
#include <pcl/registration/icp.h>      // icp��׼
#include <boost/thread/thread.hpp>
#include <pcl/common/common.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl\features\fpfh_omp.h>//omp���ټ���
#include <pcl/features/normal_3d_omp.h>
#include <pcl/visualization/pcl_visualizer.h> // ���ӻ�

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;

class Registration
{
	//���ݳ�Ա
	int m_RansacIterations;  // RANSAC�㷨������������
	float m_RansacDistance;  // RANSAC�㷨�ж��Ƿ�Ϊ�ڵ�ľ�����ֵ
//	float m_Fraction;        // RANSAC�㷨���ڵ����
	float m_MaxDistance;     // ICP����׼�ж�Ӧ���������
	int m_MaxIterations;     // ICP����������
public:
	Registration(int RansacIterations, float RansacDistance, /*float Fraction,*/ float MaxDistance, int MaxIterations) :
		m_RansacIterations(RansacIterations), m_RansacDistance(RansacDistance), /*m_Fraction(Fraction),*/ m_MaxDistance(MaxDistance), m_MaxIterations(MaxIterations) {}
	~Registration() {}

	// ��Ա����

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
