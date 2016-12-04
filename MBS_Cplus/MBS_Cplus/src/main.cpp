/*********************************************************************************
 * Copyright (c) 2016 MCG ICT
 * If you have problems about this software, please contact:
 *		chenxiaokai@ict.ac.cn
 *
 * 
 * 2016-04-20
 *********************************************************************************/
#include"MBS.hpp"
#include<unistd.h>
#include<ctime>
using namespace cv;  
using namespace std;
void Reconstruct(Mat src,Mat mask,Mat& dst) 
{ 
 /*function:morpy-Reconstruct 
 src is the source img£¬mask is label img£¬dst is the output*/ 
	Mat se=getStructuringElement(MORPH_RECT,Size(3,3)); 
	Mat tmp1(src.size(),src.type()),tmp2(src.size(),src.type()); 
	cv::min(src,mask,dst); 
	do 
	{ 
		dst.copyTo(tmp1); 
		dilate(dst,mask,se); 
		cv::min(src,mask,dst); 
		tmp2=abs(tmp1-dst); 
	}while(sum(tmp2).val[0]!=0); 
}
Mat morpySmooth(Mat I,int radius)
{
	Mat openmat,recon1,dilatemat,recon2,res;
	I=I.mul(255);
	I.convertTo(I,CV_8UC1);
	morphologyEx(I,openmat,MORPH_OPEN,Mat(radius,radius,I.type()),Point(-1,-1),1);
	Reconstruct(I,openmat,recon1);
	dilate(recon1,dilatemat,Mat(radius,radius,I.type()),Point(-1,-1),1);
	
	recon1=255-recon1;
	dilatemat=255-dilatemat;

	Reconstruct(recon1,dilatemat,recon2);

	res=255-recon2;
	return res;
}
Mat enhanceConstrast(Mat I,int b=10)
{
	I.convertTo(I,CV_32FC1);
	int total=I.rows*I.cols,num1(0),num2(0);
	double max,min,t,sum1(0),sum2(0),v1,v2;
	Point p1,p2;
	minMaxLoc(I,&min,&max,&p1,&p2);
	t=max*0.5;

	for(int i=0;i<I.rows;i++)
	{
		float* indata=I.ptr<float>(i);
		for(int j=0;j<I.cols;j++)
		{
			int temp=(*indata);
			if(temp>=t)
			{
				sum1+=temp;
				num1++;
			}
				
			else
			{
				sum2+=temp;
				num2++;
			}
				
			indata++;
		}
	}
	v1=sum1/num1;
	v2=sum2/num2;
	//cout<<t<<" "<<v1<<" "<<v2<<endl;
	for(int i=0;i<I.rows;i++)
	{
		float* indata=I.ptr<float>(i);
		for(int j=0;j<I.cols;j++)
		{
			*indata=1.0/(exp(((*indata)-0.5*(v1+v2))*(-b))+1)*255;
			indata++;
		}
	}
	//cout<<I<<endl;
	return I;
}
int main(int argc,char *argv[])
{
	if(argc!=2)
	{
		cout<<"The number of main's parameters must be 2 "<<endl;
		return 0;
	}
	if(access(argv[1],F_OK)==-1)
	{
		cout<<"The file is not exists!"<<endl;
		return 0;
	}
	Mat im=imread(argv[1]);
	Mat MBmap,morphmat,smallmat,smallres,res;
	int radius(0);
	Scalar s;
	double start,t1,t2,t3,t4;
	double scale=200.0/((im.rows>im.cols)?im.rows:im.cols);
	resize(im,smallmat,Size(im.cols*scale,im.rows*scale));


	start=clock();
	MBmap=doWork(smallmat,true,true,false);
	t1=clock();
	s =mean(MBmap);	
	radius=floor(50*sqrt(s.val[0]));	
	radius=(radius>3)?radius:3;
	morphmat=morpySmooth(MBmap,radius);	
	smallres=enhanceConstrast(morphmat);


	t4=clock();
	resize(smallres,res,Size(im.cols,im.rows));	

	cout<<"computing MBD map:  "<<(t1-start)/CLOCKS_PER_SEC<<" s"<<endl;
	cout<<"postprocessing :  "<<(t4-t1)/CLOCKS_PER_SEC<<" s"<<endl;
	cout<<"total :  "<<(t4-start)/CLOCKS_PER_SEC<<" s"<<endl;

	imwrite("../result.png",res);
	cout<<"The result has been written into result.png..."<<endl;
	return 0;
}
